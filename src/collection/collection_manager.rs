use crate::collection::collection_type::Collection;
use crate::model::model_manager::ModelManager;
use crate::model::model_utils::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::collection_utils::CollectionConfig;

pub struct CollectionManager {
    collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
    model_manager: Arc<RwLock<ModelManager>>,
    model_lookup: RwLock<HashMap<String, u32>>,
}

impl CollectionManager {
    pub fn new() -> Self {
        CollectionManager {
            collections: RwLock::new(HashMap::new()),
            model_manager: Arc::new(RwLock::new(ModelManager::new())),
            model_lookup: RwLock::new(HashMap::new()),
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
                if !lookup_guard.contains_key(requested_model.as_str()) {
                    let model_id = manager_guard
                        .load_model(requested_model.clone(), Backend::ONNX)
                        .await
                        .unwrap();
                    lookup_guard.insert(requested_model.clone(), model_id);
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
                if !lookup_guard.contains_key(requested_model.as_str()) {
                    let model_id = manager_guard
                        .load_model(requested_model.clone(), Backend::ONNX)
                        .await
                        .unwrap();
                    lookup_guard.insert(requested_model.clone(), model_id);
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
        let model_name = collection.read().await.config().model_name;
        let model_id = {
            let lookup_guard = self.model_lookup.read().await;
            lookup_guard
                .get(&model_name)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Model '{}' is not loaded", model_name))?
        };

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
}
