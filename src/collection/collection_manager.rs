use crate::collection::collection_type::Collection;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::collection_utils::CollectionConfig;

pub struct CollectionManager {
    collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
}

impl CollectionManager {
    pub fn new() -> Self {
        CollectionManager {
            collections: RwLock::new(HashMap::new()),
        }
    }

    pub async fn load_collection(&self, name: String) -> anyhow::Result<()> {
        let collection = Arc::new(RwLock::new(Collection::from(name.clone()).await.unwrap()));
        let mut collections = self.collections.write().await;
        collections.insert(name.clone(), collection);
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
}

unsafe impl Send for CollectionManager {}
unsafe impl Sync for CollectionManager {}
