use crate::model::model_utils::{ModelOutputDType, ModelTrait, ONNXModelTrait};
use anyhow;
use async_trait::async_trait;
use half::f16;
use log::info;
use ndarray::{Array2, Ix2};
use ort::{CPUExecutionProvider, GraphOptimizationLevel, Session};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::thread::available_parallelism;
use tokenizers::{PaddingParams, Tokenizer};
use tokio::task;

pub struct BertONNX {
    pub model: Arc<Session>,
    pub tokenizer: Arc<Tokenizer>,
    output_dtype: ModelOutputDType,
    output_dim: i64,
    needs_token_type_ids: bool,
}

#[async_trait]
impl ModelTrait for BertONNX {
    async fn new(model_dir: &str, model_file: &str) -> anyhow::Result<Self> {
        let model_source_path = Path::new(model_dir);
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
            .commit_from_file(model_source_path.join(model_file))
            .unwrap();

        let mut tokenizer = Tokenizer::from_file(model_source_path.join("tokenizer.json")).unwrap();

        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_to_multiple_of: None,
            pad_id: 1,
            pad_type_id: 0,
            direction: tokenizers::PaddingDirection::Right,
            pad_token: "<pad>".into(),
        }));

        // TODO: instead of using a hardcoded index,
        // use .filter to get the output tensor by name

        // determine output dtype
        let dtype = session.outputs[1]
            .output_type
            .tensor_type()
            .unwrap()
            .to_string();
        info!("Model output dtype: {:?}", dtype);

        let output_dtype = match dtype.as_str() {
            "f16" => ModelOutputDType::F16,
            "f32" => ModelOutputDType::F32,
            _ => ModelOutputDType::F32,
        };

        // determine model output dimension
        let dim = session.outputs[1]
            .output_type
            .tensor_dimensions()
            .unwrap()
            .last()
            .unwrap()
            .to_owned();
        info!("Model output dim: {dim}");

        // determine if the models needs token_type_ids
        let tti_name = "token_type_ids";
        let needs_token_type_ids = session
            .inputs
            .par_iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<&str>>()
            .contains(&tti_name);

        Ok(Self {
            model: Arc::new(session),
            tokenizer: Arc::new(tokenizer),
            output_dim: dim,
            output_dtype: output_dtype,
            needs_token_type_ids: needs_token_type_ids,
        })
    }
}

#[async_trait]
impl ONNXModelTrait for BertONNX {
    async fn predict_f16(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f16>>> {
        let output_dtype = self.output_dtype().await?;
        assert_eq!(output_dtype, ModelOutputDType::F16);

        let inputs: Vec<String> = texts.par_iter().map(|s| s.to_string()).collect();

        // Encode input strings.
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        let needs_token_type_ids = self.needs_token_type_ids;

        let (a_ids, a_mask, a_t_ids) = task::spawn_blocking(move || {
            // tokenize inputs
            let encodings = tokenizer.encode_batch(inputs.clone(), true).unwrap();
            let padded_token_length = encodings[0].len();

            // Extract token IDs and attention masks
            let ids: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_ids().iter().map(|i| *i as i64))
                .collect();
            let mask: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_attention_mask().iter().map(|i| *i as i64))
                .collect();
            let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
            let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();

            let a_t_ids = if needs_token_type_ids {
                let t_ids: Vec<i64> = encodings
                    .par_iter()
                    .flat_map_iter(|e| e.get_type_ids().iter().map(|i| *i as i64))
                    .collect();
                Some(Array2::from_shape_vec([inputs.len(), padded_token_length], t_ids).unwrap())
            } else {
                None
            };

            (a_ids, a_mask, a_t_ids)
        })
        .await?;

        // Run the model.

        let embeddings_tensor = task::spawn_blocking(move || {
            let outputs = if let Some(a_t_ids) = a_t_ids {
                model
                    .run(ort::inputs![a_ids, a_t_ids, a_mask].unwrap())
                    .unwrap()
            } else {
                model.run(ort::inputs![a_ids, a_mask].unwrap()).unwrap()
            };

            // Extract embeddings tensor.
            let embeddings_tensor = outputs[1]
                .try_extract_tensor::<f16>()
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();

            embeddings_tensor.to_owned()
        })
        .await?;

        Ok(Arc::new(embeddings_tensor.to_owned()))
    }

    async fn predict_f32(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f32>>> {
        let output_dtype = self.output_dtype().await?;
        assert_eq!(output_dtype, ModelOutputDType::F32);

        let inputs: Vec<String> = texts.par_iter().map(|s| s.to_string()).collect();

        // Encode input strings.
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        let needs_token_type_ids = self.needs_token_type_ids;

        let (a_ids, a_mask, a_t_ids) = task::spawn_blocking(move || {
            // tokenize inputs
            let encodings = tokenizer.encode_batch(inputs.clone(), true).unwrap();
            let padded_token_length = encodings[0].len();

            // Extract token IDs and attention masks
            let ids: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_ids().iter().map(|i| *i as i64))
                .collect();
            let mask: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_attention_mask().iter().map(|i| *i as i64))
                .collect();
            let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
            let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();

            let a_t_ids = if needs_token_type_ids {
                let t_ids: Vec<i64> = encodings
                    .par_iter()
                    .flat_map_iter(|e| e.get_type_ids().iter().map(|i| *i as i64))
                    .collect();
                Some(Array2::from_shape_vec([inputs.len(), padded_token_length], t_ids).unwrap())
            } else {
                None
            };

            (a_ids, a_mask, a_t_ids)
        })
        .await?;

        // Run the model.

        let embeddings_tensor = task::spawn_blocking(move || {
            let outputs = if let Some(a_t_ids) = a_t_ids {
                model
                    .run(ort::inputs![a_ids, a_t_ids, a_mask].unwrap())
                    .unwrap()
            } else {
                model.run(ort::inputs![a_ids, a_mask].unwrap()).unwrap()
            };

            // Extract embeddings tensor.
            let embeddings_tensor = outputs[1]
                .try_extract_tensor::<f32>()
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();

            embeddings_tensor.to_owned()
        })
        .await?;

        Ok(Arc::new(embeddings_tensor))
    }

    async fn output_dtype(&self) -> anyhow::Result<ModelOutputDType> {
        Ok(self.output_dtype.clone())
    }

    async fn output_dim(&self) -> anyhow::Result<i64> {
        Ok(self.output_dim)
    }
}
