use crate::model::traits::model_trait::ModelTrait;
use crate::model::traits::onnx_trait::ONNXModelTrait;
use anyhow::Error;
use async_trait::async_trait;
use half::f16;
use log::debug;
use ndarray::Array2;
use ort::{CPUExecutionProvider, GraphOptimizationLevel, Session};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct BertONNX {
    pub model: Option<Session>,
    pub tokenizer: Option<Tokenizer>,
}

#[async_trait]
impl ONNXModelTrait for BertONNX {
    //todo
}

impl BertONNX {
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            model: None,
        }
    }
}

#[async_trait]
impl ModelTrait for BertONNX {
    async fn predict(&self, input: &str) -> Result<String, String> {
        let inputs: Vec<String> = input.lines().map(|s| s.to_string()).collect();

        // Encode input strings.
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| "Model is not loaded".to_string())?;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| "Model is not loaded".to_string())?;

        let encodings = tokenizer.encode_batch(inputs.clone(), false).unwrap();
        let padded_token_length = encodings[0].len();

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
        let outputs = model.run(ort::inputs![a_ids, a_mask].unwrap()).unwrap();

        // Extract embeddings tensor.
        let embeddings_tensor = match outputs[1].try_extract_tensor::<f16>() {
            Ok(tensor) => tensor.map(|x| x.to_f32()),
            Err(e) => return Err(format!("Failed to extract tensor: {:?}", e)),
        };
        debug!("embeddings tensors: {:?}", embeddings_tensor);
        Ok("Predicted successfully".to_string())

        // let embeddings = outputs[1].try_extract_tensor::<f32>()?.into_dimensionality::<Ix2>().unwrap();
    }

    async fn load_model(&mut self, model_path: &str) -> Result<(), Error> {
        let model_source_path = Path::new(model_path);
        ort::init()
            .with_name("embedder")
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()
            .expect("Failed to initialize ORT environment");

        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(Path::join(model_source_path, "model.onnx"))
            .unwrap();

        let tokenizer =
            Tokenizer::from_file(Path::join(model_source_path, "tokenizer.json")).unwrap();
        self.model = Some(session);
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    async fn unload_model(&self) -> Result<(), String> {
        //Unload model
        Ok(())
    }
}
