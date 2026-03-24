use actix::prelude::*;
use log::info;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::ProjectError;
use crate::hf_ops::download_model;
use crate::model::backends::gemini::gemini_embedder::GeminiEmbedder;
use crate::model::backends::onnx::encoder_onnx::EncoderONNX;
use crate::model::model_utils::{Embedder, Embeddings, ModelOutputDType, ModelTrait};

// ---- Actor Definition ----
#[derive(Clone)]
pub struct ModelManagerActor {
    models: HashMap<u32, Arc<dyn Embedder>>,
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
    /// Gemini API key. Required when `path` starts with `gemini://`.
    /// Falls back to the `GEMINI_API_KEY` environment variable when `None`.
    pub gemini_api_key: Option<String>,
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
            let model: Arc<dyn Embedder> = if msg.path.starts_with("gemini://") {
                let model_name = msg.path
                    .strip_prefix("gemini://")
                    .unwrap();

                let api_key = msg
                    .gemini_api_key
                    .or_else(|| std::env::var("GEMINI_API_KEY").ok())
                    .ok_or_else(|| {
                        ProjectError::Anyhow(anyhow::anyhow!(
                            "Gemini API key not provided. \
                             Pass --gemini-api-key or set the GEMINI_API_KEY environment variable."
                        ))
                    })?;

                Arc::new(GeminiEmbedder::new(&model_name, &api_key, None))
            } else {
                let (model_dir, model_file) = if msg.path.starts_with("hf://") {
                    download_model(msg.path, msg.variant, msg.token)
                        .await
                        .map_err(|e| ProjectError::Anyhow(e))?
                } else {
                    (msg.path, msg.variant)
                };

                Arc::new(
                    EncoderONNX::new(model_dir.as_str(), model_file.as_str())
                        .map_err(|e| ProjectError::Anyhow(e))?,
                )
            };

            Ok(model)
        };

        Box::pin(actix::fut::wrap_future::<_, Self>(fut).map(
            move |result, act, _ctx| match result {
                Ok(model) => {
                    let id = act.next_id;
                    act.next_id += 1;
                    act.models.insert(id, model);
                    info!("Model loaded from {}", model_path);
                    Ok(id)
                }
                Err(e) => Err(e),
            },
        ))
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
            model
                .embed(msg.texts)
                .await
                .map_err(|e| ProjectError::Anyhow(e))
        })
    }
}

impl Handler<GetModelMetadata> for ModelManagerActor {
    type Result = Result<(i64, ModelOutputDType), ProjectError>;

    fn handle(&mut self, msg: GetModelMetadata, _ctx: &mut Context<Self>) -> Self::Result {
        let model = self
            .models
            .get(&msg.id)
            .ok_or_else(|| ProjectError::ModelNotFound(msg.id))?;

        let dim = model.output_dim().map_err(|e| ProjectError::Anyhow(e))?;
        let dtype = model.output_dtype().map_err(|e| ProjectError::Anyhow(e))?;

        Ok((dim, dtype))
    }
}

