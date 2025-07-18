use crate::actors::model_actor::{ModelManagerActor, Predict};
use crate::collection::collection_utils::{home_dir, CollectionConfig, SearchResult};
use crate::collection::vector_index::VectorIndex;
use crate::error::ProjectError;
use crate::model::model_utils::Embeddings;
use actix::prelude::*;
use anyhow::anyhow;
use duckdb::arrow::array::{PrimitiveArray, StringArray};
use duckdb::arrow::datatypes::UInt64Type;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::Connection;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use usearch::f16 as UsearchF16;

// ---- Actor Definition ----
pub struct CollectionActor {
    config: CollectionConfig,
    db_path: PathBuf,
    vector_indices: HashMap<String, Arc<RwLock<VectorIndex>>>,
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
    pub model_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<SearchResult>, ProjectError>")]
pub struct Search {
    pub column: String,
    pub query: String,
    pub limit: u32,
    pub model_id: u32,
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
                tx.execute_batch(&format!(
                    "CREATE TABLE {} AS SELECT * FROM read_json_auto('{}');",
                    collection_name, msg.path
                ))?;

                // Add _key column if it doesn't exist
                let query = format!(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{}' AND column_name = '_key';",
                    collection_name
                );
                let exists: bool = {
                    let mut stmt = tx.prepare(&query)?;
                    let count: i64 = stmt.query_row([], |row| row.get(0))?;
                    count > 0
                };

                if !exists {
                    tx.execute_batch(&format!(
                        r"CREATE SEQUENCE keys_seq;
            ALTER TABLE {} ADD COLUMN _key UBIGINT DEFAULT NEXTVAL('keys_seq');",
                        collection_name,
                    ))?;
                }

                tx.commit()?;
                Ok(())
            })
            .await?
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
                tx.execute_batch(&format!(
                    "CREATE TABLE {} AS SELECT * FROM read_parquet('{}');",
                    collection_name, msg.path
                ))?;

                // Add _key column if it doesn't exist
                let query = format!(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{}' AND column_name = '_key';",
                    collection_name
                );
                let exists: bool = {
                    let mut stmt = tx.prepare(&query)?;
                    let count: i64 = stmt.query_row([], |row| row.get(0))?;
                    count > 0
                };

                if !exists {
                    tx.execute_batch(&format!(
                        r"CREATE SEQUENCE keys_seq;
            ALTER TABLE {} ADD COLUMN _key UBIGINT DEFAULT NEXTVAL('keys_seq');",
                        collection_name,
                    ))?;
                }

                tx.commit()?;
                Ok(())
            })
            .await?
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

    fn handle(&mut self, msg: EmbedColumn, ctx: &mut Context<Self>) -> Self::Result {
        let self_addr = ctx.address();
        Box::pin(async move {
            self_addr.send(msg).await??;
            Ok(())
        })
    }
}

impl Handler<Search> for CollectionActor {
    type Result = ResponseFuture<Result<Vec<SearchResult>, ProjectError>>;

    fn handle(&mut self, msg: Search, _ctx: &mut Context<Self>) -> Self::Result {
        let vector_index = self.vector_indices.get(&msg.column).cloned();
        let model_manager = self.model_manager.clone();
        let db_path = self.db_path.clone();
        let collection_name = self.config.name.clone();

        Box::pin(async move {
            let vector_index = vector_index.ok_or_else(|| {
                ProjectError::Anyhow(anyhow!("Vector index for column '{}' not found", msg.column))
            })?;

            let column_name = msg.column.clone();
            let query_embedding = model_manager
                .send(Predict {
                    id: msg.model_id,
                    texts: vec![msg.query],
                })
                .await??;

            let similarity_results = tokio::task::spawn_blocking({
                let index_clone = vector_index.clone();
                move || match query_embedding {
                    Embeddings::F16(emb) => index_clone
                        .read()
                        .map_err(|e| ProjectError::Anyhow(anyhow!(e.to_string())))?
                        .search::<UsearchF16>(
                            emb.as_ptr() as *const UsearchF16,
                            emb.dim().1,
                            msg.limit as usize,
                        ),
                    Embeddings::F32(emb) => index_clone
                        .read()
                        .map_err(|e| ProjectError::Anyhow(anyhow!(e.to_string())))?
                        .search::<f32>(emb.as_ptr(), emb.dim().1, msg.limit as usize),
                }
            })
            .await??;

            let keys: Vec<u64> = similarity_results.iter().map(|r| r.key).collect();
            if keys.is_empty() {
                return Ok(Vec::new());
            }

            let contents: Vec<String> = tokio::task::spawn_blocking({
                let db_path = db_path.clone();
                let collection_name = collection_name.clone();
                let column_name = column_name.clone();
                move || -> Result<_, ProjectError> {
                    let conn = Connection::open(&db_path)?;
                    let keys_str = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(", ");
                    let query = format!(
                        "SELECT _key, {} FROM {} WHERE _key IN ({});",
                        column_name, collection_name, keys_str
                    );
                    let mut stmt = conn.prepare(&query)?;
                    let rbs: Vec<RecordBatch> = stmt.query_arrow([])?.collect();
                    let rb = rbs.first().ok_or_else(|| {
                        ProjectError::Anyhow(anyhow!("No records found"))
                    })?;
                    let key_array = rb
                        .column_by_name("_key")
                        .ok_or_else(|| {
                            ProjectError::Anyhow(anyhow!("Column '_key' not found"))
                        })?
                        .as_any()
                        .downcast_ref::<PrimitiveArray<UInt64Type>>()
                        .ok_or_else(|| {
                            ProjectError::Anyhow(anyhow!("_key is not of type UInt64"))
                        })?;
                    let text_array = rb
                        .column_by_name(&column_name)
                        .ok_or_else(|| {
                            ProjectError::Anyhow(anyhow!(
                                "Column '{}' not found",
                                column_name
                            ))
                        })?
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            ProjectError::Anyhow(anyhow!("Column is not of type String"))
                        })?;
                    let mut content_map = key_array
                        .iter()
                        .zip(text_array.iter())
                        .filter_map(|(k, v)| k.map(|k_val| (k_val, v.map(|v_val| v_val.to_string()))))
                        .filter_map(|(k, v)| v.map(|v_val| (k, v_val)))
                        .collect::<HashMap<_, _>>();
                    let ordered_contents =
                        keys.iter().map(|k| content_map.remove(k).unwrap()).collect();
                    Ok(ordered_contents)
                }
            })
            .await??;

            let search_results = similarity_results
                .into_iter()
                .zip(contents.into_iter())
                .map(|(sim, content)| SearchResult {
                    content,
                    key: sim.key,
                    score: sim.score,
                })
                .collect();

            Ok(search_results)
        })
    }
}