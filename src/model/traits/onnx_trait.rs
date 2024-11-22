use super::model_trait::ModelTrait;
use async_trait::async_trait;

#[allow(dead_code)]
#[async_trait]
pub trait ONNXModelTrait: ModelTrait {}
