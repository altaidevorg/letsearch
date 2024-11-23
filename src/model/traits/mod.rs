pub mod model_trait;
pub mod onnx_trait;

pub enum Backend {
    ONNX,
}

#[derive(Clone)]
pub enum ModelOutputDType {
    F32,
    F16,
    Int8,
}
