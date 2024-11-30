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
    #[serde(default = "default_db_path")]
    pub db_path: String,
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
    String::from("minilm")
}

fn default_db_path() -> String {
    String::from("data.db")
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
            db_path: default_db_path(),
            serialization_version: default_serialization_version(),
        }
    }
}
