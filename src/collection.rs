use crate::model::manager::ModelManager;
use crate::model::model_utils::Embeddings;
use crate::vector_index::VectorIndex;
use anyhow;
use duckdb::arrow::array::StringArray;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::Connection;
use log::{debug, info};
use std::fs;
use std::path::Path;
use std::time::Instant;
use usearch::{IndexOptions, MetricKind, ScalarKind};

pub struct Collection {
    name: String,
    conn: Connection,
    vector_index: Option<VectorIndex>,
}

impl Collection {
    pub fn new(name: String, overwrite: bool) -> anyhow::Result<Self> {
        debug!("creating new Collection instance");
        let collection_dir = Path::new(&name);
        if overwrite && collection_dir.exists() {
            debug!("Collection already exists, overwriting");
            fs::remove_dir_all(name.as_str())?;
        }

        fs::create_dir_all(name.as_str())?;
        let db_path = collection_dir.join("data.db");
        let conn = Connection::open(db_path)?;
        debug!("Connection opened to DB");

        Ok(Collection {
            name: name,
            conn: conn,
            vector_index: None,
        })
    }

    pub fn import_jsonl(&self, jsonl_path: &str) -> anyhow::Result<()> {
        self.conn.execute_batch(
            format!(
                "CREATE TABLE {} AS SELECT * FROM read_json_auto('{}');",
                &self.name, jsonl_path
            )
            .as_str(),
        )?;
        debug!("JSONL file imported");

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
                column_name, &self.name, batch_size, offset
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
            let index_path = String::from("test1/index");
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
