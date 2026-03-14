use crate::model::model_utils::{ModelOutputDType, ModelTrait, ONNXModelTrait};
use anyhow;
use half::f16;
use log::info;
use ndarray::{Array2, Ix2};
#[cfg(feature = "cuda")]
use ort::CUDAExecutionProvider;
use ort::{CPUExecutionProvider, GraphOptimizationLevel, Session};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::sync::Once;
use std::thread::available_parallelism;
use tokenizers::{PaddingParams, Tokenizer};

static ORT_INIT: Once = Once::new();

pub struct EncoderONNX {
    pub model: Arc<Session>,
    pub tokenizer: Arc<Tokenizer>,
    output_dtype: ModelOutputDType,
    output_dim: i64,
    needs_token_type_ids: bool,
}

impl ModelTrait for EncoderONNX {
    fn new(model_dir: &str, model_file: &str) -> anyhow::Result<Self> {
        ORT_INIT.call_once(|| {
            ort::init()
                .with_name("onnx_model")
                .with_execution_providers([
                    #[cfg(feature = "cuda")]
                    CUDAExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ])
                .commit()
                .expect("Failed to initialize ORT environment");
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
            .outputs
            .iter()
            .position(|o| o.name == "sentence_embedding")
            .unwrap_or_else(|| if session.outputs.len() > 1 { 1 } else { 0 });

        // determine output dtype
        let dtype = session.outputs[output_idx]
            .output_type
            .tensor_type()
            .ok_or_else(|| anyhow::anyhow!("Coult not determine output tensor type"))?
            .to_string();
        info!("Model output dtype: {:?}", dtype);

        let output_dtype = match dtype.as_str() {
            "f16" => ModelOutputDType::F16,
            "f32" => ModelOutputDType::F32,
            _ => ModelOutputDType::F32,
        };

        // determine model output dimension
        let dim = session.outputs[output_idx]
            .output_type
            .tensor_dimensions()
            .ok_or_else(|| anyhow::anyhow!("Coult not determine tensor dimensions"))?
            .last()
            .ok_or_else(|| anyhow::anyhow!("Tensor has no dimensions"))?
            .to_owned();
        info!("Model output dim: {dim}");

        // determine if the models needs token_type_ids
        let tti_name = "token_type_ids";
        let needs_token_type_ids = session
            .inputs
            .par_iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<&str>>()
            .contains(&tti_name);

        Ok(Self {
            model: Arc::new(session),
            tokenizer: Arc::new(tokenizer),
            output_dim: dim,
            output_dtype: output_dtype,
            needs_token_type_ids: needs_token_type_ids,
        })
    }
}

impl ONNXModelTrait for EncoderONNX {
    fn predict_f16(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f16>>> {
        let output_dtype = self.output_dtype()?;
        assert_eq!(output_dtype, ModelOutputDType::F16);

        let inputs: Vec<String> = texts.par_iter().map(|s| s.to_string()).collect();

        // Encode input strings.
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        let needs_token_type_ids = self.needs_token_type_ids;

        let (a_ids, a_mask, a_t_ids) = {
            // tokenize inputs
            let encodings = tokenizer
                .encode_batch(inputs.clone(), true)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let padded_token_length = encodings[0].len();

            // Extract token IDs and attention masks
            let ids: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_ids().iter().map(|i| *i as i64))
                .collect();
            let mask: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_attention_mask().iter().map(|i| *i as i64))
                .collect();
            let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            let a_t_ids = if needs_token_type_ids {
                let t_ids: Vec<i64> = encodings
                    .par_iter()
                    .flat_map_iter(|e| e.get_type_ids().iter().map(|i| *i as i64))
                    .collect();
                Some(
                    Array2::from_shape_vec([inputs.len(), padded_token_length], t_ids)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                )
            } else {
                None
            };

            (a_ids, a_mask, a_t_ids)
        };

        // Run the model.

        let embeddings_tensor = {
            let outputs = if let Some(a_t_ids) = a_t_ids {
                model
                    .run(
                        ort::inputs![a_ids, a_t_ids, a_mask]
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    )
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?
            } else {
                model
                    .run(ort::inputs![a_ids, a_mask].map_err(|e| anyhow::anyhow!(e.to_string()))?)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?
            };

            // Extract embeddings tensor.
            let embeddings_tensor = outputs[1]
                .try_extract_tensor::<f16>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .into_dimensionality::<Ix2>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            embeddings_tensor.to_owned()
        };

        Ok(Arc::new(embeddings_tensor.to_owned()))
    }

    fn predict_f32(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f32>>> {
        let output_dtype = self.output_dtype()?;
        assert_eq!(output_dtype, ModelOutputDType::F32);

        let inputs: Vec<String> = texts.par_iter().map(|s| s.to_string()).collect();

        // Encode input strings.
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        let needs_token_type_ids = self.needs_token_type_ids;

        let (a_ids, a_mask, a_t_ids) = {
            // tokenize inputs
            let encodings = tokenizer
                .encode_batch(inputs.clone(), true)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let padded_token_length = encodings[0].len();

            // Extract token IDs and attention masks
            let ids: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_ids().iter().map(|i| *i as i64))
                .collect();
            let mask: Vec<i64> = encodings
                .par_iter()
                .flat_map_iter(|e| e.get_attention_mask().iter().map(|i| *i as i64))
                .collect();
            let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            let a_t_ids = if needs_token_type_ids {
                let t_ids: Vec<i64> = encodings
                    .par_iter()
                    .flat_map_iter(|e| e.get_type_ids().iter().map(|i| *i as i64))
                    .collect();
                Some(
                    Array2::from_shape_vec([inputs.len(), padded_token_length], t_ids)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                )
            } else {
                None
            };

            (a_ids, a_mask, a_t_ids)
        };

        // Run the model.

        let embeddings_tensor = {
            let outputs = if let Some(a_t_ids) = a_t_ids {
                model
                    .run(
                        ort::inputs![a_ids, a_t_ids, a_mask]
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    )
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?
            } else {
                model
                    .run(ort::inputs![a_ids, a_mask].map_err(|e| anyhow::anyhow!(e.to_string()))?)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?
            };

            // Extract embeddings tensor.
            let embeddings_tensor = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .into_dimensionality::<Ix2>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            embeddings_tensor.to_owned()
        };

        Ok(Arc::new(embeddings_tensor))
    }

    fn output_dtype(&self) -> anyhow::Result<ModelOutputDType> {
        Ok(self.output_dtype.clone())
    }

    fn output_dim(&self) -> anyhow::Result<i64> {
        Ok(self.output_dim)
    }
}
