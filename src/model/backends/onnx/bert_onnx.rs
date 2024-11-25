use crate::model::model_utils::{ModelOutputDType, ModelTrait, ONNXModelTrait};
use anyhow;
use async_trait::async_trait;
use half::f16;
use log::{debug, info};
use ndarray::{Array2, Ix2};
use ort::{CPUExecutionProvider, GraphOptimizationLevel, Session};
use rayon::prelude::*;
use std::sync::Arc;
use std::thread::available_parallelism;
use std::{path::Path, time::Instant};
use tokenizers::{PaddingParams, Tokenizer};

pub struct BertONNX {
    pub model: Option<Session>,
    pub tokenizer: Option<Tokenizer>,
    output_dtype: Option<ModelOutputDType>,
    needs_token_type_ids: Option<bool>,
}

impl BertONNX {
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            model: None,
            output_dtype: None,
            needs_token_type_ids: None,
        }
    }
}

#[async_trait]
impl ModelTrait for BertONNX {
    async fn load_model(&mut self, model_path: &str) -> anyhow::Result<()> {
        let model_source_path = Path::new(model_path);
        ort::init()
            .with_name("onnx_model")
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()
            .expect("Failed to initialize ORT environment");

        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(available_parallelism()?.get())
            .unwrap()
            .commit_from_file(Path::join(model_source_path, "model.onnx"))
            .unwrap();

        let mut tokenizer =
            Tokenizer::from_file(Path::join(model_source_path, "tokenizer.json")).unwrap();

        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_to_multiple_of: None,
            pad_id: 1,
            pad_type_id: 0,
            direction: tokenizers::PaddingDirection::Right,
            pad_token: "<pad>".into(),
        }));

        // determine output dtype
        let dtype = session.outputs[0]
            .output_type
            .tensor_type()
            .unwrap()
            .to_string();
        info!("output dtype: {:?}", dtype);
        self.output_dtype = match dtype.as_str() {
            "f16" => Some(ModelOutputDType::F16),
            "f32" => Some(ModelOutputDType::F32),
            _ => None,
        };

        // determine if the models needs token_type_ids
        let tti_name = "token_type_ids";
        let needs_token_type_ids = session
            .inputs
            .par_iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<&str>>()
            .contains(&tti_name);
        self.needs_token_type_ids = Some(needs_token_type_ids);

        self.model = Some(session);
        self.tokenizer = Some(tokenizer);

        Ok(())
    }

    async fn unload_model(&self) -> anyhow::Result<()> {
        //Unload model
        Ok(())
    }
}

#[async_trait]
impl ONNXModelTrait for BertONNX {
    async fn predict_f16(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f16>>> {
        let inputs: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // Encode input strings.
        let model = self.model.as_ref().unwrap();
        let tokenizer = self.tokenizer.as_ref().unwrap();

        // tokenize inputs
        let encodings = tokenizer.encode_batch(inputs.clone(), true).unwrap();
        let padded_token_length = encodings[0].len();

        // Extract token IDs and attention masks
        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
        let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();

        // Run the model.
        let start = Instant::now();
        let outputs = if self.needs_token_type_ids.unwrap() {
            let t_ids = encodings
                .iter()
                .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
                .collect();
            let a_t_ids =
                Array2::from_shape_vec([inputs.len(), padded_token_length], t_ids).unwrap();
            model
                .run(ort::inputs![a_ids, a_t_ids, a_mask].unwrap())
                .unwrap()
        } else {
            model.run(ort::inputs![a_ids, a_mask].unwrap()).unwrap()
        };

        debug!("actual inference took: {:?}", start.elapsed());

        // Extract embeddings tensor.
        let embeddings_tensor = outputs[1]
            .try_extract_tensor::<f16>()?
            .into_dimensionality::<Ix2>()
            .unwrap();

        Ok(Arc::new(embeddings_tensor.to_owned()))
    }

    async fn predict_f32(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f32>>> {
        let inputs: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // Encode input strings.
        let model = self.model.as_ref().unwrap();
        let tokenizer = self.tokenizer.as_ref().unwrap();

        // tokenize inputs
        let encodings = tokenizer.encode_batch(inputs.clone(), true).unwrap();
        let padded_token_length = encodings[0].len();

        // Extract token IDs and attention masks
        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
        let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();

        // Run the model.
        let start = Instant::now();
        let outputs = if self.needs_token_type_ids.unwrap() {
            let t_ids = encodings
                .iter()
                .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
                .collect();
            let a_t_ids =
                Array2::from_shape_vec([inputs.len(), padded_token_length], t_ids).unwrap();
            model
                .run(ort::inputs![a_ids, a_t_ids, a_mask].unwrap())
                .unwrap()
        } else {
            model.run(ort::inputs![a_ids, a_mask].unwrap()).unwrap()
        };

        debug!("actual inference took: {:?}", start.elapsed());

        // Extract embeddings tensor.
        let embeddings_tensor = outputs[1]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix2>()
            .unwrap();

        Ok(Arc::new(embeddings_tensor.to_owned()))
    }

    async fn output_dtype(&self) -> ModelOutputDType {
        self.output_dtype.as_ref().unwrap().clone()
    }
}
