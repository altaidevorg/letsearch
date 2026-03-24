use crate::model::model_utils::{Embedder, Embeddings, ModelOutputDType};
use async_trait::async_trait;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Default output dimension for Gemini embedding models.
/// `gemini-embedding-2-preview` supports dimensions from 256 to 3072.
const DEFAULT_OUTPUT_DIM: i64 = 3072;

/// Embedding client for Google Gemini embedding models.
///
/// Supports any model accessible through the Gemini API, such as
/// `gemini-embedding-2-preview`. The output is always f32. The default
/// embedding dimension for `gemini-embedding-2-preview` is 3072, but
/// callers can request a smaller dimension (256–3072) via `output_dim`.
pub struct GeminiEmbedder {
    model_name: String,
    api_key: String,
    output_dim: i64,
    client: reqwest::Client,
}

impl GeminiEmbedder {
    /// Create a new `GeminiEmbedder`.
    ///
    /// * `model_name` – the model identifier, e.g. `"gemini-embedding-2-preview"`.
    /// * `api_key`    – a valid Gemini API key.
    /// * `output_dim` – optional embedding dimension override (256–3072).
    ///   Defaults to 3072 when `None`.
    pub fn new(model_name: &str, api_key: &str, output_dim: Option<i64>) -> Self {
        Self {
            model_name: model_name.to_string(),
            api_key: api_key.to_string(),
            output_dim: output_dim.unwrap_or(DEFAULT_OUTPUT_DIM),
            client: reqwest::Client::new(),
        }
    }
}

// ---------- Serde helpers for the Gemini batchEmbedContents API ----------

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    content: Content,
}

#[derive(Serialize)]
struct BatchEmbedRequest {
    requests: Vec<EmbedRequest>,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<EmbeddingValues>,
}

// -------------------------------------------------------------------------

#[async_trait]
impl Embedder for GeminiEmbedder {
    fn output_dim(&self) -> anyhow::Result<i64> {
        Ok(self.output_dim)
    }

    fn output_dtype(&self) -> anyhow::Result<ModelOutputDType> {
        Ok(ModelOutputDType::F32)
    }

    async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Embeddings> {
        let model_full = format!("models/{}", self.model_name);

        let requests: Vec<EmbedRequest> = texts
            .iter()
            .map(|text| EmbedRequest {
                model: model_full.clone(),
                content: Content {
                    parts: vec![Part { text: text.clone() }],
                },
            })
            .collect();

        let body = BatchEmbedRequest { requests };

        let url = format!(
            "{}/{}:batchEmbedContents?key={}",
            GEMINI_API_BASE, self.model_name, self.api_key
        );

        let response: BatchEmbedResponse = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Gemini API request failed: {}", e))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("Gemini API returned an error: {}", e))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse Gemini API response: {}", e))?;

        let n = response.embeddings.len();
        let dim = self.output_dim as usize;

        let mut result = Array2::<f32>::zeros((n, dim));
        for (i, emb) in response.embeddings.iter().enumerate() {
            if emb.values.len() != dim {
                return Err(anyhow::anyhow!(
                    "Gemini API returned embedding dimension {} for item {}, expected {}",
                    emb.values.len(),
                    i,
                    dim
                ));
            }
            for (j, &v) in emb.values.iter().enumerate() {
                result[[i, j]] = v;
            }
        }

        Ok(Embeddings::F32(Arc::new(result)))
    }
}
