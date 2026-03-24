use async_trait::async_trait;
use crate::model::model_utils::{Embedder, Embeddings, ModelOutputDType, ModelTrait, ONNXModelTrait};
use half::f16;
use log::info;
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
#[cfg(feature = "cuda")]
use ort::CUDAExecutionProvider;
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::path::Path;
use std::sync::Arc;
use std::sync::Once;
use std::thread::available_parallelism;
use tokenizers::{PaddingParams, Tokenizer};

static ORT_INIT: Once = Once::new();

/// A lock-free wrapper around `ort::Session` that provides interior mutability.
///
/// # Safety
///
/// This is safe because:
/// 1. ONNX Runtime's C API (`OrtRun`) is inherently thread-safe — the underlying
///    `OrtSession` handles concurrent inference internally.
/// 2. `Session` is already marked `Send + Sync` by the `ort` crate.
/// 3. The `&mut self` requirement on `Session::run` in ort v2 is a Rust-level
///    API constraint to prevent output lifetime aliasing, not because the session
///    is genuinely mutated.
/// 4. All callers (`predict_f16`, `predict_f32`) fully consume outputs into owned
///    `Array2` within the same scope, so there is no aliasing of output borrows.
struct SyncUnsafeSession(UnsafeCell<Session>);

// SAFETY: OrtSession is thread-safe at the C level. Session is already Send + Sync.
unsafe impl Send for SyncUnsafeSession {}
unsafe impl Sync for SyncUnsafeSession {}

impl SyncUnsafeSession {
    fn new(session: Session) -> Self {
        Self(UnsafeCell::new(session))
    }

    /// Get a mutable reference to the inner session for calling `run`.
    ///
    /// # Safety
    ///
    /// This is safe because ONNX Runtime handles thread safety at the C level,
    /// and all callers consume the outputs within the same scope (no aliasing).
    fn get_mut(&self) -> &mut Session {
        unsafe { &mut *self.0.get() }
    }
}

pub struct EncoderONNX {
    pub tokenizer: Arc<Tokenizer>,
    model: Arc<SyncUnsafeSession>,
    pub needs_token_type_ids: bool,
    pub output_dtype: ModelOutputDType,
    pub output_dim: i64,
}

impl ModelTrait for EncoderONNX {
    fn new(model_dir: &str, model_file: &str) -> anyhow::Result<Self> {
        ORT_INIT.call_once(|| {
            let _ = ort::init().with_name("onnx_model").commit();
        });

        let model_source_path = Path::new(model_dir);

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_intra_threads(available_parallelism()?.get())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(model_source_path.join(model_file))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut tokenizer = Tokenizer::from_file(model_source_path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_to_multiple_of: None,
            pad_id: 1,
            pad_type_id: 0,
            direction: tokenizers::PaddingDirection::Right,
            pad_token: "<pad>".into(),
        }));

        // determine output index dynamically
        let output_idx = session
            .outputs()
            .iter()
            .position(|o| o.name() == "sentence_embedding")
            .unwrap_or_else(|| if session.outputs().len() > 1 { 1 } else { 0 });

        // determine output dtype
        let dtype = session.outputs()[output_idx]
            .dtype()
            .tensor_type()
            .ok_or_else(|| anyhow::anyhow!("Could not determine output tensor type"))?
            .to_string();
        info!("Model output dtype: {:?}", dtype);

        let output_dtype = match dtype.as_str() {
            "f16" => ModelOutputDType::F16,
            "f32" => ModelOutputDType::F32,
            _ => ModelOutputDType::F32,
        };

        // determine model output dimension
        let dim = session.outputs()[output_idx]
            .dtype()
            .tensor_shape()
            .ok_or_else(|| anyhow::anyhow!("Could not determine tensor dimensions"))?
            .last()
            .ok_or_else(|| anyhow::anyhow!("Tensor has no dimensions"))?
            .to_owned();
        info!("Model output dim: {dim}");

        // determine if the models needs token_type_ids
        let tti_name = "token_type_ids";
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .map(|i| i.name())
            .collect::<Vec<&str>>()
            .contains(&tti_name);

        Ok(Self {
            model: Arc::new(SyncUnsafeSession::new(session)),
            tokenizer: Arc::new(tokenizer),
            output_dim: dim,
            output_dtype,
            needs_token_type_ids,
        })
    }
}

impl ONNXModelTrait for EncoderONNX {
    fn predict_f16(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f16>>> {
        assert_eq!(self.output_dtype, ModelOutputDType::F16);
        run_predict_f16(&self.model, &self.tokenizer, self.needs_token_type_ids, texts)
    }

    fn predict_f32(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f32>>> {
        assert_eq!(self.output_dtype, ModelOutputDType::F32);
        run_predict_f32(&self.model, &self.tokenizer, self.needs_token_type_ids, texts)
    }

    fn output_dtype(&self) -> anyhow::Result<ModelOutputDType> {
        Ok(self.output_dtype.clone())
    }

    fn output_dim(&self) -> anyhow::Result<i64> {
        Ok(self.output_dim)
    }
}

fn run_predict_f16(
    model: &SyncUnsafeSession,
    tokenizer: &Tokenizer,
    needs_token_type_ids: bool,
    texts: Vec<&str>,
) -> anyhow::Result<Arc<Array2<f16>>> {
    let inputs: Vec<String> = texts.par_iter().map(|s| s.to_string()).collect();

    let (ids, mask, a_t_ids, batch_len, token_len) = {
        let encodings = tokenizer
            .encode_batch(inputs.clone(), true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let padded_token_length = encodings[0].len();

        let ids: Vec<i64> = encodings
            .par_iter()
            .flat_map_iter(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .par_iter()
            .flat_map_iter(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        let a_t_ids = if needs_token_type_ids {
            let t_ids: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_type_ids().iter().map(|i| *i as i64))
                .collect();
            Some(t_ids)
        } else {
            None
        };

        (ids, mask, a_t_ids, inputs.len(), padded_token_length)
    };

    let embeddings_tensor = {
        let shape = [batch_len, token_len];
        let session = model.get_mut();

        let outputs = if let Some(a_t_ids) = a_t_ids {
            session
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array((shape, ids)).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    "token_type_ids" => Tensor::from_array((shape, a_t_ids)).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    "attention_mask" => Tensor::from_array((shape, mask.clone())).map_err(|e| anyhow::anyhow!(e.to_string()))?
                ])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        } else {
            session
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array((shape, ids)).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    "attention_mask" => Tensor::from_array((shape, mask)).map_err(|e| anyhow::anyhow!(e.to_string()))?
                ])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        };

        let (output_shape, output_data) = outputs[1]
            .try_extract_tensor::<f16>()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        ndarray::ArrayView2::from_shape(
            (output_shape[0] as usize, output_shape[1] as usize),
            output_data,
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .to_owned()
    };

    Ok(Arc::new(embeddings_tensor))
}

fn run_predict_f32(
    model: &SyncUnsafeSession,
    tokenizer: &Tokenizer,
    needs_token_type_ids: bool,
    texts: Vec<&str>,
) -> anyhow::Result<Arc<Array2<f32>>> {
    let inputs: Vec<String> = texts.par_iter().map(|s| s.to_string()).collect();

    let (ids, mask, a_t_ids, batch_len, token_len) = {
        let encodings = tokenizer
            .encode_batch(inputs.clone(), true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let padded_token_length = encodings[0].len();

        let ids: Vec<i64> = encodings
            .par_iter()
            .flat_map_iter(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .par_iter()
            .flat_map_iter(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        let a_t_ids = if needs_token_type_ids {
            let t_ids: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_type_ids().iter().map(|i| *i as i64))
                .collect();
            Some(t_ids)
        } else {
            None
        };

        (ids, mask, a_t_ids, inputs.len(), padded_token_length)
    };

    let embeddings_tensor = {
        let shape = [batch_len, token_len];
        let session = model.get_mut();

        let outputs = if let Some(a_t_ids) = a_t_ids {
            session
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array((shape, ids)).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    "token_type_ids" => Tensor::from_array((shape, a_t_ids)).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    "attention_mask" => Tensor::from_array((shape, mask.clone())).map_err(|e| anyhow::anyhow!(e.to_string()))?
                ])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        } else {
            session
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array((shape, ids)).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    "attention_mask" => Tensor::from_array((shape, mask)).map_err(|e| anyhow::anyhow!(e.to_string()))?
                ])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        };

        let (output_shape, output_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        ndarray::ArrayView2::from_shape(
            (output_shape[0] as usize, output_shape[1] as usize),
            output_data,
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .to_owned()
    };

    Ok(Arc::new(embeddings_tensor))
}

#[async_trait]
impl Embedder for EncoderONNX {
    fn output_dim(&self) -> anyhow::Result<i64> {
        Ok(self.output_dim)
    }

    fn output_dtype(&self) -> anyhow::Result<ModelOutputDType> {
        Ok(self.output_dtype.clone())
    }

    async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Embeddings> {
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let dtype = self.output_dtype.clone();
        let needs_token_type_ids = self.needs_token_type_ids;

        tokio::task::spawn_blocking(move || {
            let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            match dtype {
                ModelOutputDType::F16 => {
                    let result = run_predict_f16(&model, &tokenizer, needs_token_type_ids, texts_ref)?;
                    Ok(Embeddings::F16(result))
                }
                ModelOutputDType::F32 => {
                    let result = run_predict_f32(&model, &tokenizer, needs_token_type_ids, texts_ref)?;
                    Ok(Embeddings::F32(result))
                }
                ModelOutputDType::Int8 => {
                    unimplemented!("int8 dynamic quantization not yet implemented")
                }
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("Spawn blocking error: {}", e))?
    }
}
