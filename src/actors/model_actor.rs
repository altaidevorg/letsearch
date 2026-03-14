use actix::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;
use log::info;

use crate::error::ProjectError;
use crate::model::model_utils::{Embeddings, ModelOutputDType, ONNXModel, ModelTrait};
use crate::model::backends::onnx::bert_onnx::BertONNX;
use crate::hf_ops::download_model;

// ---- Actor Definition ----
#[derive(Clone)]
pub struct ModelManagerActor {
    models: HashMap<u32, Arc<dyn ONNXModel>>,
    next_id: u32,
}

impl ModelManagerActor {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_id: 1,
        }
    }
}

impl Actor for ModelManagerActor {
    type Context = Context<Self>;
}

// ---- Message Definitions ----
#[derive(Message)]
#[rtype(result = "Result<u32, ProjectError>")]
pub struct LoadModel {
    pub path: String,
    pub variant: String,
    pub token: Option<String>,
}

#[derive(Message)]
#[rtype(result = "Result<Embeddings, ProjectError>")]
pub struct Predict {
    pub id: u32,
    pub texts: Vec<String>,
}

#[derive(Message)]
#[rtype(result = "Result<(i64, ModelOutputDType), ProjectError>")]
pub struct GetModelMetadata {
    pub id: u32,
}

// ---- Message Handlers ----
impl Handler<LoadModel> for ModelManagerActor {
    type Result = ResponseActFuture<Self, Result<u32, ProjectError>>;

    fn handle(&mut self, msg: LoadModel, _ctx: &mut Context<Self>) -> Self::Result {
        let model_path = msg.path.clone();
        
        let fut = async move {
            let (model_dir, model_file) = if msg.path.starts_with("hf://") {
                download_model(msg.path, msg.variant, msg.token)
                    .await
                    .map_err(|e| ProjectError::Anyhow(e))?
            } else {
                (msg.path, msg.variant)
            };

            let model: Arc<dyn ONNXModel> = Arc::new(
                BertONNX::new(model_dir.as_str(), model_file.as_str())
                    .map_err(|e| ProjectError::Anyhow(e))?
            );
            Ok(model)
        };

        Box::pin(
            actix::fut::wrap_future::<_, Self>(fut)
                .map(move |result, act, _ctx| {
                    match result {
                        Ok(model) => {
                            let id = act.next_id;
                            act.next_id += 1;
                            act.models.insert(id, model);
                            info!("Model loaded from {}", model_path);
                            Ok(id)
                        }
                        Err(e) => Err(e),
                    }
                })
        )
    }
}

impl Handler<Predict> for ModelManagerActor {
    type Result = ResponseFuture<Result<Embeddings, ProjectError>>;

    fn handle(&mut self, msg: Predict, _ctx: &mut Context<Self>) -> Self::Result {
        let model = match self.models.get(&msg.id) {
            Some(m) => m.clone(),
            None => return Box::pin(async move { Err(ProjectError::ModelNotFound(msg.id)) }),
        };
        
        Box::pin(async move {
            // Calculate embeddings in a blocking task since it's CPU-bound
            tokio::task::spawn_blocking(move || -> Result<Embeddings, ProjectError> {
                let dtype = model.output_dtype().map_err(|e| ProjectError::Anyhow(e))?;
                
                let texts_str: Vec<&str> = msg.texts.iter().map(|s| s.as_str()).collect();
                
                match dtype {
                    ModelOutputDType::F16 => {
                        let result = model.predict_f16(texts_str).map_err(|e| ProjectError::Anyhow(e))?;
                        Ok(Embeddings::F16(result))
                    }
                    ModelOutputDType::F32 => {
                        let result = model.predict_f32(texts_str).map_err(|e| ProjectError::Anyhow(e))?;
                        Ok(Embeddings::F32(result))
                    }
                    ModelOutputDType::Int8 => {
                        unimplemented!("int8 dynamic quantization not yet implemented")
                    }
                }
            })
            .await
            .map_err(|e| ProjectError::Anyhow(anyhow::anyhow!("Spawn blocking error: {}", e)))?
        })
    }
}

impl Handler<GetModelMetadata> for ModelManagerActor {
    type Result = Result<(i64, ModelOutputDType), ProjectError>;

    fn handle(&mut self, msg: GetModelMetadata, _ctx: &mut Context<Self>) -> Self::Result {
        let model = self.models.get(&msg.id).ok_or_else(|| ProjectError::ModelNotFound(msg.id))?;

        let dim = model.output_dim().map_err(|e| ProjectError::Anyhow(e))?;
        let dtype = model.output_dtype().map_err(|e| ProjectError::Anyhow(e))?;
        
        Ok((dim, dtype))
    }
}
