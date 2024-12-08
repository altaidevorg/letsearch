use crate::collection::collection_type::Collection;
use crate::model::model_manager::ModelManager;
use crate::model::model_utils::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::collection_utils::{CollectionConfig, SearchResult};

pub struct CollectionManager {
    collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
    model_manager: Arc<RwLock<ModelManager>>,
    model_lookup: RwLock<HashMap<(String, String), u32>>,
    token: Option<String>,
}

impl CollectionManager {
    pub fn new(token: Option<String>) -> Self {
        CollectionManager {
            collections: RwLock::new(HashMap::new()),
            model_manager: Arc::new(RwLock::new(ModelManager::new())),
            model_lookup: RwLock::new(HashMap::new()),
            token: token,
        }
    }

    pub async fn load_collection(&self, name: String) -> anyhow::Result<()> {
        let collection = Arc::new(RwLock::new(Collection::from(name.clone()).await.unwrap()));
        let collection_guard = collection.read().await;
        let requested_models = collection_guard.requested_models().await;
        if !requested_models.is_empty() {
            let manager_guard = self.model_manager.write().await;
            for requested_model in requested_models {
                let mut lookup_guard = self.model_lookup.write().await;
                if !lookup_guard.contains_key(&requested_model) {
                    let (model_path, model_variant) = requested_model.clone();
                    let model_id = manager_guard
                        .load_model(
                            model_path.clone(),
                            model_variant.clone(),
                            Backend::ONNX,
                            self.token.clone(),
                        )
                        .await
                        .unwrap();
                    lookup_guard.insert(requested_model, model_id);
                }
            }
        }

        let mut collections = self.collections.write().await;
        collections.insert(name.clone(), collection.clone());

        Ok(())
    }

    pub async fn create_collection(
        &self,
        config: CollectionConfig,
        overwrite: bool,
    ) -> anyhow::Result<()> {
        let name = config.name.clone();
        let collection = Arc::new(RwLock::new(Collection::new(config, overwrite).await?));
        let collection_guard = collection.read().await;
        let requested_models = collection_guard.requested_models().await;
        if !requested_models.is_empty() {
            let manager_guard = self.model_manager.write().await;
            for requested_model in requested_models {
                let mut lookup_guard = self.model_lookup.write().await;
                if !lookup_guard.contains_key(&requested_model) {
                    let (model_path, model_variant) = requested_model.clone();
                    let model_id = manager_guard
                        .load_model(
                            model_path.clone(),
                            model_variant.clone(),
                            Backend::ONNX,
                            self.token.clone(),
                        )
                        .await
                        .unwrap();
                    lookup_guard.insert(requested_model, model_id);
                }
            }
        }

        let mut collections = self.collections.write().await;
        collections.insert(name.clone(), collection.clone());

        Ok(())
    }

    pub async fn get_collections(&self) -> Vec<String> {
        let collections = self.collections.read().await;
        let collection_names: Vec<String> = collections.keys().cloned().collect();

        collection_names
    }

    pub async fn get_collection_configs(&self) -> Vec<CollectionConfig> {
        let collections = self.collections.read().await;
        let mut configs = Vec::new();
        for collection in collections.values() {
            let collection = collection.read().await;
            configs.push(collection.config());
        }

        configs
    }

    pub async fn get_collection_config(
        &self,
        collection_name: String,
    ) -> anyhow::Result<CollectionConfig> {
        let collection = self
            .collections
            .read()
            .await
            .get(collection_name.as_str())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Collection '{}' does not exist", collection_name))?;

        let config = collection.read().await.config();
        Ok(config)
    }

    pub async fn import_jsonl(
        &self,
        collection_name: &str,
        jsonl_path: &str,
    ) -> anyhow::Result<()> {
        // Acquire a read lock on the collections map
        let collection = {
            let collections_guard = self.collections.read().await;

            match collections_guard.get(collection_name) {
                Some(collection) => collection.clone(),
                None => {
                    return Err(anyhow::anyhow!(
                        "Collection '{}' does not exist",
                        collection_name
                    ));
                }
            }
        };

        // Acquire a write lock on the collection and call import_jsonl
        let collection_guard = collection.write().await;
        collection_guard.import_jsonl(jsonl_path).await
    }

    pub async fn import_parquet(
        &self,
        collection_name: &str,
        parquet_path: &str,
    ) -> anyhow::Result<()> {
        // Acquire a read lock on the collections map
        let collection = {
            let collections_guard = self.collections.read().await;

            match collections_guard.get(collection_name) {
                Some(collection) => collection.clone(),
                None => {
                    return Err(anyhow::anyhow!(
                        "Collection '{}' does not exist",
                        collection_name
                    ));
                }
            }
        };

        // Acquire a write lock on the collection and call import_jsonl
        let collection_guard = collection.write().await;
        collection_guard.import_parquet(parquet_path).await
    }

    pub async fn embed_column(
        &self,
        collection_name: &str,
        column_name: &str,
        batch_size: u64,
    ) -> anyhow::Result<()> {
        // Fetch collection
        let collection = {
            let collections_guard = self.collections.read().await;
            collections_guard
                .get(collection_name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Collection '{}' does not exist", collection_name))?
        };

        // Fetch model ID
        let config = collection.read().await.config();
        let model = (config.model_name, config.model_variant);

        let model_id = self
            .model_lookup
            .read()
            .await
            .get(&model)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Model '{:?}' is not loaded", model))?;

        // Perform embedding
        let mut collection_guard = collection.write().await;
        collection_guard
            .embed_column(
                column_name,
                batch_size,
                self.model_manager.clone(),
                model_id,
            )
            .await
    }

    pub async fn search(
        &self,
        collection_name: String,
        column_name: String,
        query: String,
        limit: u32,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let collection = self
            .collections
            .read()
            .await
            .get(collection_name.as_str())
            .cloned()
            .ok_or_else(|| {
                return anyhow::anyhow!("Collection '{}' does not exist", collection_name);
            })?;
        let config = collection.read().await.config();
        let model = (config.model_name, config.model_variant);

        let model_id = self
            .model_lookup
            .read()
            .await
            .get(&model)
            .copied()
            .ok_or_else(|| {
                return anyhow::anyhow!(
                    "Model requested by collection is not loaded. This should never happen"
                );
            })?;

        let results = collection
            .read()
            .await
            .search(
                column_name,
                query,
                limit,
                self.model_manager.clone(),
                model_id,
            )
            .await?;

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::CollectionManager;
    use crate::collection::collection_utils::{home_dir, CollectionConfig};

    #[tokio::test]
    async fn test_collection_manager() {
        let manager = CollectionManager::new(None);
        let mut config = CollectionConfig::default();
        config.name = String::from("test_collection");

        config.model_name = "hf://mys/minilm".to_string();
        config.model_variant = "i8".to_string();
        config.index_columns = vec![String::from("context")];

        manager
            .create_collection(config.clone(), true)
            .await
            .unwrap();
        assert_eq!(
            manager.get_collections().await,
            vec!["test_collection".to_string()]
        );

        assert_eq!(manager.get_collection_configs().await[0], config);
        manager
            .import_jsonl(
                "test_collection",
                "hf://datasets/llmware/rag_instruct_benchmark_tester/*.jsonl",
            )
            .await
            .unwrap();

        let column_name = "context";
        let batch_size = 32;
        manager
            .embed_column("test_collection", column_name, batch_size)
            .await
            .unwrap();

        // Search
        let query = "What is the total amount of the invoice?".to_string();
        let results = manager
            .search(
                "test_collection".to_string(),
                column_name.to_string(),
                query,
                10,
            )
            .await
            .unwrap();
        assert!(!results.is_empty()); // This might not always be true, depending on the data and query

        fs::remove_dir_all(home_dir().join("models").join("mys").join("minilm")).unwrap();
    }
}
