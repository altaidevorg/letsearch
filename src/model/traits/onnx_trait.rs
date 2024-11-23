use super::model_trait::ModelTrait;
use super::{Backend, ModelOutputDType};
use anyhow;
use async_trait::async_trait;
use half::f16;
use ndarray::Array2;
use std::sync::Arc;

#[async_trait]
pub trait ONNXModelTrait: ModelTrait {
    async fn output_dtype(&self) -> ModelOutputDType;
    async fn predict_f16(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f16>>>;

    fn backend(&self) -> Backend {
        Backend::ONNX
    }
}

pub trait ONNXModel: ModelTrait + ONNXModelTrait + Send + Sync {}
impl<T> ONNXModel for T where T: ModelTrait + ONNXModelTrait + Send + Sync {}
