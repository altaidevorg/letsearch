use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

const DEFAULT_HOME_DIR: &str = ".letsearch";

pub fn home_dir() -> PathBuf {
    std::env::var("LETSEARCH_HOME")
        .unwrap_or_else(|_| DEFAULT_HOME_DIR.to_string())
        .into()
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
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
    String::from("hf://mys/minilm")
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

    pub fn from_file(name: &str) -> anyhow::Result<Self> {
        let collection_dir = home_dir().join("collections").join(name);
        let config_path = collection_dir.join("config.json");
        let config_file = File::open(&config_path).map_err(|e| {
            anyhow::anyhow!(
                "Cannot open collection config at {}: {}. \
                 Run `letsearch index` from the same working directory (or set LETSEARCH_HOME) \
                 so the collection is created under .letsearch/collections/<name>/.",
                config_path.display(),
                e
            )
        })?;
        let config: CollectionConfig = serde_json::from_reader(config_file)?;
        Ok(config)
    }

    /// Writes this config to `LETSEARCH_HOME/collections/<name>/config.json`.
    /// Called when a collection is opened so `search`, `serve`, and `add-docs` can reload it in a new process.
    pub fn write_to_collection_dir(&self) -> anyhow::Result<()> {
        let collection_dir = home_dir().join("collections").join(self.name.as_str());
        std::fs::create_dir_all(&collection_dir)?;
        let path = collection_dir.join("config.json");
        let f = File::create(&path)?;
        let w = BufWriter::new(f);
        serde_json::to_writer_pretty(w, self)?;
        Ok(())
    }

    /// Create collection directory, write `config.json`, and an empty DuckDB file. No embedding model is loaded.
    pub fn init_on_disk(config: &CollectionConfig, overwrite: bool) -> anyhow::Result<PathBuf> {
        let collection_dir = home_dir().join("collections").join(config.name.as_str());
        if collection_dir.exists() {
            if !overwrite {
                anyhow::bail!(
                    "Collection '{}' already exists at {}. Pass --overwrite to delete and recreate.",
                    config.name,
                    collection_dir.display()
                );
            }
            std::fs::remove_dir_all(&collection_dir).with_context(|| {
                format!(
                    "Failed to remove existing collection directory {}",
                    collection_dir.display()
                )
            })?;
        }
        std::fs::create_dir_all(&collection_dir).with_context(|| {
            format!("Failed to create {}", collection_dir.display())
        })?;
        config.write_to_collection_dir()?;
        let db_path = collection_dir.join(config.db_path.as_str());
        duckdb::Connection::open(&db_path).with_context(|| {
            format!(
                "Failed to create empty database {}",
                db_path.display()
            )
        })?;
        Ok(collection_dir)
    }
}

#[derive(Serialize)]
pub struct SearchResult {
    pub content: String,
    pub key: u64,
    pub score: f32,
}
