use actix::prelude::*;
use std::sync::Arc;
use crate::model::model_manager::ModelManager;
use crate::model::model_utils::{Embeddings, ModelOutputDType};
use crate::error::ProjectError;

// ---- Actor Definition ----
pub struct ModelManagerActor {
    manager: Arc<ModelManager>,
}

impl ModelManagerActor {
    pub fn new() -> Self {
        Self {
            manager: Arc::new(ModelManager::new()),
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
    type Result = ResponseFuture<Result<u32, ProjectError>>;

    fn handle(&mut self, msg: LoadModel, _ctx: &mut Context<Self>) -> Self::Result {
        let manager = self.manager.clone();
        Box::pin(async move {
            manager
                .load_model(msg.path, msg.variant, crate::model::model_utils::Backend::ONNX, msg.token)
                .await
                .map_err(|_| ProjectError::ModelNotFound(0))
        })
    }
}

impl Handler<Predict> for ModelManagerActor {
    type Result = ResponseFuture<Result<Embeddings, ProjectError>>;

    fn handle(&mut self, msg: Predict, _ctx: &mut Context<Self>) -> Self::Result {
        let manager = self.manager.clone();
        Box::pin(async move {
            let texts_str: Vec<&str> = msg.texts.iter().map(|s| s.as_str()).collect();
            manager
                .predict(msg.id, texts_str)
                .await
                .map_err(|_| ProjectError::ModelNotFound(msg.id))
        })
    }
}

impl Handler<GetModelMetadata> for ModelManagerActor {
    type Result = ResponseFuture<Result<(i64, ModelOutputDType), ProjectError>>;

    fn handle(&mut self, msg: GetModelMetadata, _ctx: &mut Context<Self>) -> Self::Result {
        let manager = self.manager.clone();
        Box::pin(async move {
            let dim = manager.output_dim(msg.id).await.map_err(|_| ProjectError::ModelNotFound(msg.id))?;
            let dtype = manager.output_dtype(msg.id).await.map_err(|_| ProjectError::ModelNotFound(msg.id))?;
            Ok((dim, dtype))
        })
    }
}
