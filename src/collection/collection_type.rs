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
use std::fs;
use std::fs::File;
use std::time::Instant;
use usearch::{IndexOptions, MetricKind, ScalarKind};

pub struct Collection {
    config: CollectionConfig,
    conn: Connection,
    vector_index: Option<VectorIndex>,
}

impl Collection {
    pub fn new(config: CollectionConfig, overwrite: bool) -> anyhow::Result<Self> {
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
            conn: conn,
            vector_index: None,
        })
    }

    pub fn from(name: String) -> anyhow::Result<Self> {
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
        let vector_index = VectorIndex::from(index_path.to_path_buf()).unwrap();

        Ok(Collection {
            config: config,
            conn: conn,
            vector_index: Some(vector_index),
        })
    }

    pub fn config(&self) -> CollectionConfig {
        self.config.clone()
    }

    pub fn import_jsonl(&self, jsonl_path: &str) -> anyhow::Result<()> {
        let start = Instant::now();
        self.conn.execute_batch(
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

    pub fn get_single_column(
        &self,
        column_name: &str,
        batch_size: u64,
        offset: u64,
    ) -> anyhow::Result<Vec<String>> {
        assert!(batch_size >= 1);
        let mut stmt = self.conn.prepare(
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

    pub async fn embed_column_with_offset(
        &mut self,
        column_name: &str,
        batch_size: u64,
        offset: u64,
        model_manager: &ModelManager,
        model_id: u32,
    ) -> anyhow::Result<()> {
        let start = Instant::now();
        let texts = self
            .get_single_column(column_name, batch_size, offset)
            .unwrap();
        debug!("getting texts from DB took: {:?}", start.elapsed());
        let start = Instant::now();
        let inputs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = model_manager.predict(model_id, inputs).await.unwrap();

        let index = self.vector_index.get_or_insert_with(|| {
            let index_path = home_dir()
                .join("collections")
                .join(self.config.name.as_str())
                .join("index")
                .join(column_name);
            let options = IndexOptions {
                dimensions: 384,
                metric: MetricKind::Cos,
                quantization: ScalarKind::F32,
                connectivity: 0,
                expansion_add: 0,
                expansion_search: 0,
                multi: true,
            };
            let mut index = VectorIndex::new(index_path, true).unwrap();
            index.with_options(&options, 20000).unwrap();
            index
        });

        index.save()?;

        match embeddings {
            Embeddings::F16(emb) => debug!("output shape: {:?}", emb.dim()),
            Embeddings::F32(emb) => {
                let (num_vectors, vector_dim) = emb.dim();
                let ids: Vec<_> = (offset..offset + num_vectors as u64).collect();
                index.add(&ids, emb.as_ptr(), vector_dim).await.unwrap();

                debug!("output shape: {:?}", emb.dim());
            }
        }

        info!("Embedding texts took: {:?}", start.elapsed());
        Ok(())
    }

    pub async fn embed_column(
        &mut self,
        column_name: &str,
        batch_size: u64,
        model_manager: &ModelManager,
        model_id: u32,
    ) -> anyhow::Result<()> {
        let num_batches = 2048 / batch_size;
        info!("Starting to index column '{column_name}' in batches of {batch_size}");

        let start = Instant::now();
        for batch in 0..num_batches {
            self.embed_column_with_offset(
                column_name,
                batch_size,
                batch * batch_size,
                model_manager,
                model_id,
            )
            .await
            .unwrap();
        }
        self.vector_index.as_ref().unwrap().save().unwrap();

        info!("Total duration: {:?}", start.elapsed());

        Ok(())
    }
}
