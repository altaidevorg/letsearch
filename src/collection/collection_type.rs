use crate::collection::collection_utils::{home_dir, CollectionConfig};
use crate::collection::vector_index::VectorIndex;
use crate::model::model_manager::ModelManager;
use crate::model::model_utils::Embeddings;
use anyhow::Error;
use duckdb::arrow::array::StringArray;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::Connection;
use log::{debug, info};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use usearch::{IndexOptions, MetricKind, ScalarKind};

pub struct Collection {
    config: CollectionConfig,
    conn: Arc<RwLock<Connection>>,
    vector_index: RwLock<HashMap<String, Arc<RwLock<VectorIndex>>>>,
}

impl Collection {
    pub async fn new(config: CollectionConfig, overwrite: bool) -> anyhow::Result<Self> {
        debug!("creating new Collection instance");
        let name = config.name.as_str();
        let collection_dir = home_dir().join("collections").join(name);
        let collection_dir_str = collection_dir.to_str().unwrap();
        if overwrite && collection_dir.exists() {
            debug!("Collection already exists, overwriting");
            fs::remove_dir_all(collection_dir_str)?;
            debug!("removed existing collection for overwriting");
        }

        fs::create_dir_all(collection_dir_str)?;
        debug!("Created collection dir: {collection_dir_str}");
        let db_path = collection_dir.join(config.db_path.as_str());

        let conn = Connection::open(db_path).expect("error while trying to open connection to db");
        debug!("Connection opened to DB");

        let config_file = File::create(collection_dir.join("config.json").to_str().unwrap())
            .expect("error while trying to create config.json");
        let _ = serde_json::to_writer(config_file, &config).unwrap();

        Ok(Collection {
            config: config,
            conn: Arc::new(RwLock::new(conn)),
            vector_index: RwLock::new(HashMap::new()),
        })
    }

    pub async fn from(name: String) -> anyhow::Result<Self> {
        let collection_dir = home_dir().join("collections").join(name.as_str());
        if !collection_dir.exists() {
            return Err(Error::msg("Collection {name} does not exist"));
        }

        let config_path = collection_dir.join("config.json");
        if !config_path.exists() {
            return Err(Error::msg("config file does not exist"));
        }

        let config_file = File::open(config_path).unwrap();
        let config: CollectionConfig = serde_json::from_reader(config_file)?;
        let conn = Connection::open(collection_dir.join(config.db_path.as_str()))?;
        let index_path = collection_dir
            .join("index")
            .join(config.index_columns[0].as_str());
        let vector_indexes = RwLock::new(HashMap::new());
        let vector_index = VectorIndex::from(index_path.to_path_buf())?;
        {
            let mut indexes_guard = vector_indexes.write().await;
            indexes_guard.insert(name.clone(), Arc::new(RwLock::new(vector_index)));
        }

        Ok(Collection {
            config: config,
            conn: Arc::new(RwLock::new(conn)),
            vector_index: vector_indexes,
        })
    }

    pub fn config(&self) -> CollectionConfig {
        self.config.clone()
    }

    pub async fn import_jsonl(&self, jsonl_path: &str) -> anyhow::Result<()> {
        let start = Instant::now();
        let conn = self.conn.clone();
        let conn_guard = conn.write().await;
        conn_guard.execute_batch(
            format!(
                "CREATE TABLE {} AS SELECT * FROM read_json_auto('{}');",
                &self.config.name, jsonl_path
            )
            .as_str(),
        )?;
        info!(
            "Records imported from {:?} in {:?}",
            jsonl_path,
            start.elapsed()
        );

        Ok(())
    }

    pub async fn get_single_column(
        &self,
        column_name: &str,
        batch_size: u64,
        offset: u64,
    ) -> anyhow::Result<Vec<String>> {
        assert!(batch_size >= 1);
        let conn = self.conn.clone();
        let conn_guard = conn.read().await;
        let mut stmt = conn_guard.prepare(
            format!(
                "SELECT {} FROM {} LIMIT {} OFFSET {};",
                column_name, &self.config.name, batch_size, offset
            )
            .as_str(),
        )?;
        let result: Vec<RecordBatch> = stmt.query_arrow([])?.collect();
        assert_eq!(result.len(), 1);
        let batch = &result[0];
        //let num_rows = batch.num_rows();
        //let num_cols = batch.num_columns();

        let schema = batch.schema();
        let column_names: Vec<&str> = schema
            .fields
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<&str>>();
        let col = &column_names[0];
        let col_array = batch
            .column_by_name(col)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let col_values: Vec<String> = col_array
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect::<Vec<String>>();

        Ok(col_values)
    }

    async fn embed_column_with_offset(
        &mut self,
        column_name: &str,
        batch_size: u64,
        offset: u64,
        model_manager: Arc<RwLock<ModelManager>>,
        model_id: u32,
    ) -> anyhow::Result<()> {
        let start = Instant::now();
        let texts = self
            .get_single_column(column_name, batch_size, offset)
            .await
            .unwrap();
        debug!("getting texts from DB took: {:?}", start.elapsed());
        let start = Instant::now();
        let inputs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = model_manager
            .read()
            .await
            .predict(model_id, inputs)
            .await
            .unwrap();

        match embeddings {
            Embeddings::F16(emb) => debug!("output shape: {:?}", emb.dim()),
            Embeddings::F32(emb) => {
                let (num_vectors, vector_dim) = emb.dim();
                let ids: Vec<_> = (offset..offset + num_vectors as u64).collect();
                let indexes_guard = self.vector_index.read().await;
                let index = indexes_guard.get(column_name).unwrap().clone();
                let index_guard = index.write().await;
                index_guard
                    .add(&ids, emb.as_ptr(), vector_dim)
                    .await
                    .unwrap();

                debug!("output shape: {:?}", emb.dim());
            }
        }

        debug!("Embedding texts took: {:?}", start.elapsed());
        Ok(())
    }

    pub async fn embed_column(
        &mut self,
        column_name: &str,
        batch_size: u64,
        model_manager: Arc<RwLock<ModelManager>>,
        model_id: u32,
    ) -> anyhow::Result<()> {
        let num_batches = 4096 / batch_size;
        info!("Starting to index column '{column_name}' in batches of {batch_size}");

        {
            let mut indexes_guard = self.vector_index.write().await;
            if !indexes_guard.contains_key(column_name) {
                let vector_dim = model_manager
                    .read()
                    .await
                    .output_dim(model_id)
                    .await
                    .unwrap();

                let index_path = home_dir()
                    .join("collections")
                    .join(self.config.name.as_str())
                    .join("index")
                    .join(column_name);
                let options = IndexOptions {
                    dimensions: vector_dim as usize,
                    metric: MetricKind::Cos,
                    quantization: ScalarKind::F32,
                    connectivity: 0,
                    expansion_add: 0,
                    expansion_search: 0,
                    multi: true,
                };
                let mut index = VectorIndex::new(index_path, true).unwrap();
                index.with_options(&options, 20000).unwrap();
                indexes_guard.insert(column_name.to_string(), Arc::new(RwLock::new(index)));
            }
        }

        let start = Instant::now();

        for batch in 0..num_batches {
            let elapsed = start.elapsed();
            let steps_completed = batch as f64;
            let total_steps = num_batches as f64;
            let eta = if steps_completed > 0.0 {
                elapsed.mul_f64((total_steps - steps_completed) / steps_completed)
            } else {
                Duration::ZERO
            };

            // Format ETA as seconds

            // print progress
            print!("\r{} / {} batches - ETA: {:?}", batch, total_steps, eta);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            self.embed_column_with_offset(
                column_name,
                batch_size,
                batch * batch_size,
                model_manager.clone(),
                model_id,
            )
            .await
            .unwrap();
        }

        // save index to disk
        self.vector_index
            .read()
            .await
            .clone()
            .get(column_name)
            .unwrap()
            .read()
            .await
            .save()
            .unwrap();

        println!("");
        info!("Total duration: {:?}", start.elapsed());

        Ok(())
    }

    pub async fn requested_models(&self) -> Vec<String> {
        vec![self.config.model_name.clone()]
    }
}

// Needed because Rust does not understand Collection::conn is managed for thread safety.
unsafe impl Send for Collection {}
unsafe impl Sync for Collection {}
