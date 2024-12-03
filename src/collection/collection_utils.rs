use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const DEFAULT_HOME_DIR: &str = ".letsearch";

pub fn home_dir() -> PathBuf {
    std::env::var("LETSEARCH_HOME")
        .unwrap_or_else(|_| DEFAULT_HOME_DIR.to_string())
        .into()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CollectionConfig {
    #[serde(default = "default_collection_name")]
    pub name: String,
    #[serde(default = "default_index_columns")]
    pub index_columns: Vec<String>,
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_model_variant")]
    pub model_variant: String,
    #[serde(default = "default_db_path")]
    pub db_path: String,
    #[serde(default = "default_index_dir")]
    pub index_dir: String,
    #[serde(default = "default_serialization_version")]
    pub serialization_version: u32,
}

fn default_collection_name() -> String {
    String::from("default")
}

fn default_index_columns() -> Vec<String> {
    vec![String::from("text")]
}

fn default_model_name() -> String {
    String::from("mys/minilm")
}

fn default_model_variant() -> String {
    String::from("f32")
}

fn default_db_path() -> String {
    String::from("data.db")
}

fn default_index_dir() -> String {
    String::from("index")
}

fn default_serialization_version() -> u32 {
    1
}

impl CollectionConfig {
    pub fn default() -> Self {
        CollectionConfig {
            name: default_collection_name(),
            index_columns: default_index_columns(),
            model_name: default_model_name(),
            model_variant: default_model_variant(),
            db_path: default_db_path(),
            index_dir: default_index_dir(),
            serialization_version: default_serialization_version(),
        }
    }
}

#[derive(Serialize)]
pub struct SearchResult {
    pub content: String,
    pub key: u64,
    pub score: f32,
}
