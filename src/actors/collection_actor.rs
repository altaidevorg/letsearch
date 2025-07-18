use actix::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use crate::collection::collection_utils::{CollectionConfig, SearchResult, home_dir};
use crate::collection::vector_index::VectorIndex;
use crate::actors::model_actor::{ModelManagerActor, GetModelMetadata, Predict};
use crate::error::ProjectError;
use crate::model::model_utils::{Embeddings, ModelOutputDType};
use duckdb::Connection;
use usearch::{IndexOptions, MetricKind, ScalarKind};
use crate::collection::collection_type::Collection;
use duckdb::arrow::array::{PrimitiveArray, StringArray};
use duckdb::arrow::datatypes::UInt64Type;
use duckdb::arrow::record_batch::RecordBatch;
use usearch::f16 as UsearchF16;

// ---- Actor Definition ----
pub struct CollectionActor {
    config: CollectionConfig,
    db_path: PathBuf,
    vector_indices: HashMap<String, VectorIndex>,
    model_manager: Addr<ModelManagerActor>,
}

impl CollectionActor {
    pub fn new(config: CollectionConfig, model_manager: Addr<ModelManagerActor>) -> Self {
        let db_path = home_dir()
            .join("collections")
            .join(config.name.as_str())
            .join(config.db_path.as_str());
        Self {
            config,
            db_path,
            vector_indices: HashMap::new(),
            model_manager,
        }
    }
}

impl Actor for CollectionActor {
    type Context = Context<Self>;
}

// ---- Message Definitions ----
#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct ImportJsonl {
    pub path: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct ImportParquet {
    pub path: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct EmbedColumn {
    pub name: String,
    pub batch_size: u64,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<SearchResult>, ProjectError>")]
pub struct Search {
    pub column: String,
    pub query: String,
    pub limit: u32,
}

#[derive(Message)]
#[rtype(result = "Result<CollectionConfig, ProjectError>")]
pub struct GetConfig;

// ---- Message Handlers ----
impl Handler<ImportJsonl> for CollectionActor {
    type Result = ResponseFuture<Result<(), ProjectError>>;

    fn handle(&mut self, msg: ImportJsonl, _ctx: &mut Context<Self>) -> Self::Result {
        let db_path = self.db_path.clone();
        let collection_name = self.config.name.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || -> Result<(), ProjectError> {
                let mut conn = Connection::open(&db_path)?;
                let tx = conn.transaction()?;
                tx.execute_batch(
                    &format!(
                        "CREATE TABLE {} AS SELECT * FROM read_json_auto('{}');",
                        collection_name, msg.path
                    )
                )?;
                tx.commit()?;
                Ok(())
            }).await.unwrap()
        })
    }
}

impl Handler<ImportParquet> for CollectionActor {
    type Result = ResponseFuture<Result<(), ProjectError>>;

    fn handle(&mut self, msg: ImportParquet, _ctx: &mut Context<Self>) -> Self::Result {
        let db_path = self.db_path.clone();
        let collection_name = self.config.name.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || -> Result<(), ProjectError> {
                let mut conn = Connection::open(&db_path)?;
                let tx = conn.transaction()?;
                tx.execute_batch(
                    &format!(
                        "CREATE TABLE {} AS SELECT * FROM read_parquet('{}');",
                        collection_name, msg.path
                    )
                )?;
                tx.commit()?;
                Ok(())
            }).await.unwrap()
        })
    }
}

impl Handler<GetConfig> for CollectionActor {
    type Result = Result<CollectionConfig, ProjectError>;

    fn handle(&mut self, _msg: GetConfig, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(self.config.clone())
    }
}

impl Handler<EmbedColumn> for CollectionActor {
    type Result = ResponseFuture<Result<(), ProjectError>>;

    fn handle(&mut self, msg: EmbedColumn, _ctx: &mut Context<Self>) -> Self::Result {
        let db_path = self.db_path.clone();
        let collection_name = self.config.name.clone();
        let model_manager = self.model_manager.clone();
        let batch_size = msg.batch_size;
        let column_name = msg.name;

        Box::pin(async move {
            let count: u64 = tokio::task::spawn_blocking(move || {
                let conn = Connection::open(&db_path).unwrap();
                let query = format!("SELECT COUNT('{}') FROM {};", column_name, collection_name);
                let mut stmt = conn.prepare(&query).unwrap();
                let count: i64 = stmt.query_row([], |row| row.get(0)).unwrap();
                count as u64
            }).await.unwrap();

            let num_batches = (count + batch_size - 1) / batch_size;

            for batch in 0..num_batches {
                // get data from db
                // get embeddings
                // add to index
            }

            Ok(())
        })
    }
}
