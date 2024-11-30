use super::model_utils::{Backend, Embeddings, ModelOutputDType, ONNXModel};
use crate::model::backends::onnx::bert_onnx::BertONNX;
use anyhow::Error;
use half::f16;
use ndarray::Array2;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ModelManager {
    models: RwLock<HashMap<u32, Arc<RwLock<dyn ONNXModel>>>>,
    next_id: RwLock<u32>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            next_id: RwLock::new(1),
        }
    }

    pub async fn load_model(&self, model_path: String, model_type: Backend) -> anyhow::Result<u32> {
        let model: Arc<RwLock<dyn ONNXModel>> = match model_type {
            Backend::ONNX => Arc::new(RwLock::new(BertONNX::new())),
            // _ => unreachable!("not implemented"),
        };

        {
            let mut model_guard = model.write().await;
            model_guard
                .load_model(&model_path)
                .await
                .map_err(|e| Error::msg(e.to_string()))?;
        }

        let mut next_id = self.next_id.write().await;
        let model_id = *next_id;
        *next_id += 1;

        let mut models = self.models.write().await;
        models.insert(model_id, model);

        Ok(model_id)
    }

    pub async fn predict_f16(
        &self,
        model_id: u32,
        texts: Vec<&str>,
    ) -> anyhow::Result<Arc<Array2<f16>>> {
        let models = self.models.read().await;
        match models.get(&model_id) {
            Some(model) => {
                let model_guard = model.read().await; // Lock the RwLock for reading
                Ok(model_guard.predict_f16(texts).await?)
            }
            None => Err(Error::msg("Model not found")),
        }
    }

    pub async fn predict_f32(
        &self,
        model_id: u32,
        texts: Vec<&str>,
    ) -> anyhow::Result<Arc<Array2<f32>>> {
        let models = self.models.read().await;
        match models.get(&model_id) {
            Some(model) => {
                let model_guard = model.read().await; // Lock the RwLock for reading
                Ok(model_guard.predict_f32(texts).await?)
            }
            None => Err(Error::msg("Model not found")),
        }
    }

    pub async fn predict(&self, model_id: u32, texts: Vec<&str>) -> anyhow::Result<Embeddings> {
        let output_dtype = self.output_dtype(model_id).await?;
        match output_dtype {
            ModelOutputDType::F16 => Ok(Embeddings::F16(
                self.predict_f16(model_id, texts).await.unwrap().to_owned(),
            )),
            ModelOutputDType::F32 => Ok(Embeddings::F32(
                self.predict_f32(model_id, texts).await.unwrap().to_owned(),
            )),
            ModelOutputDType::Int8 => {
                unimplemented!("int8 dynamic quantization not yet implemented")
            }
        }
    }

    pub async fn output_dtype(&self, model_id: u32) -> anyhow::Result<ModelOutputDType> {
        let models = self.models.read().await;
        match models.get(&model_id) {
            Some(model) => {
                let model_guard = model.read().await; // Lock the RwLock for reading
                model_guard.output_dtype().await
            }
            None => Err(Error::msg("Model not loaded")),
        }
    }

    pub async fn output_dim(&self, model_id: u32) -> anyhow::Result<i64> {
        let models = self.models.read().await;
        match models.get(&model_id) {
            Some(model) => {
                let model_guard = model.read().await; // Lock the RwLock for reading
                model_guard.output_dim().await
            }
            None => Err(Error::msg("Model not loaded")),
        }
    }
}
