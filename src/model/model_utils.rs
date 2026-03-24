use anyhow;
use async_trait::async_trait;
use half::f16;
use ndarray::Array2;
use std::sync::Arc;

pub enum Backend {
    ONNX,
    Gemini,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelOutputDType {
    F32,
    F16,
    #[allow(dead_code)]
    Int8,
}

pub enum Embeddings {
    F16(Arc<Array2<f16>>),
    F32(Arc<Array2<f32>>),
}

/// General async embedding trait implemented by all model backends.
#[async_trait]
pub trait Embedder: Send + Sync {
    fn output_dim(&self) -> anyhow::Result<i64>;
    fn output_dtype(&self) -> anyhow::Result<ModelOutputDType>;
    async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Embeddings>;
}

pub trait ModelTrait {
    fn new(model_dir: &str, model_file: &str) -> anyhow::Result<Self>
    where
        Self: Sized;
}

pub trait ONNXModelTrait: ModelTrait {
    fn output_dtype(&self) -> anyhow::Result<ModelOutputDType>;
    fn output_dim(&self) -> anyhow::Result<i64>;
    fn predict_f16(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f16>>>;
    fn predict_f32(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f32>>>;

    #[allow(dead_code)]
    fn backend(&self) -> Backend {
        Backend::ONNX
    }
}

pub trait ONNXModel: ModelTrait + ONNXModelTrait + Send + Sync {}
impl<T> ONNXModel for T where T: ModelTrait + ONNXModelTrait + Send + Sync {}
