use actix::prelude::*;
use anyhow::anyhow;
use duckdb::arrow::array::{PrimitiveArray, StringArray};
use duckdb::arrow::datatypes::UInt64Type;
use duckdb::arrow::record_batch::RecordBatch;
use log::info;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use usearch::f16 as UsearchF16;
use usearch::{IndexOptions, MetricKind, ScalarKind};

use crate::actors::model_actor::{GetModelMetadata, ModelManagerActor, Predict};
use crate::collection::collection_utils::{home_dir, CollectionConfig, SearchResult};
use crate::collection::vector_index::VectorIndex;
use crate::error::ProjectError;
use crate::model::model_utils::{Embeddings, ModelOutputDType};

// ---- Db Messages ----

#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbImportJsonl {
    pub path: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbImportParquet {
    pub path: String,
}

#[derive(Message)]
#[rtype(result = "Result<u64, ProjectError>")]
pub struct DbGetRowCount {
    pub column: String,
}

#[derive(Message)]
#[rtype(result = "Result<bool, ProjectError>")]
pub struct DbCheckIndex {
    pub column: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbInitIndex {
    pub column: String,
    pub dimensions: usize,
    pub quantization: ScalarKind,
}

#[derive(Message)]
#[rtype(result = "Result<(Vec<String>, Vec<u64>), ProjectError>")]
pub struct DbGetBatch {
    pub column: String,
    pub batch_size: u64,
    pub offset: u64,
}

#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbAddEmbeddings {
    pub column: String,
    pub keys: Vec<u64>,
    pub embeddings: Embeddings,
}

#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbSaveIndex {
    pub column: String,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<SearchResult>, ProjectError>")]
pub struct DbSearchAndFetch {
    pub column: String,
    pub query_embedding: Embeddings,
    pub limit: usize,
}

// ---- CollectionDbActor (SyncActor) ----

pub struct CollectionDbActor {
    conn: duckdb::Connection,
    vector_indices: HashMap<String, VectorIndex>,
    config: CollectionConfig,
}

impl CollectionDbActor {
    pub fn new(config: CollectionConfig) -> Self {
        let collection_dir = home_dir()
            .join("collections")
            .join(config.name.as_str());
            
        // ensure dir exists
        std::fs::create_dir_all(&collection_dir).unwrap();
        
        let db_path = collection_dir.join(config.db_path.as_str());
        let conn = duckdb::Connection::open(&db_path).expect("Failed to open DuckDB connection");
            
        let mut vector_indices = HashMap::new();
        let index_dir = collection_dir.join(config.index_dir.as_str());
        if index_dir.exists() && !config.index_columns.is_empty() {
            for index_column in config.index_columns.iter() {
                let index_path = index_dir.join(index_column.as_str());
                if let Ok(vector_index) = VectorIndex::from(index_path.to_path_buf()) {
                    vector_indices.insert(index_column.clone(), vector_index);
                }
            }
        }
        
        Self {
            conn,
            vector_indices,
            config,
        }
    }
}

impl Actor for CollectionDbActor {
    type Context = SyncContext<Self>;
}

impl Handler<DbImportJsonl> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbImportJsonl, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let tx = self.conn.transaction()?;
        tx.execute_batch(&format!(
            "CREATE TABLE {} AS SELECT * FROM read_json_auto('{}');",
            self.config.name, msg.path
        ))?;

        let query = format!(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{}' AND column_name = '_key';",
            self.config.name
        );
        let mut stmt = tx.prepare(&query)?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        if count == 0 {
            tx.execute_batch(&format!(
                r"CREATE SEQUENCE keys_seq;
    ALTER TABLE {} ADD COLUMN _key UBIGINT DEFAULT NEXTVAL('keys_seq');",
                self.config.name,
            ))?;
        }
        tx.commit()?;
        Ok(())
    }
}

impl Handler<DbImportParquet> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbImportParquet, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let tx = self.conn.transaction()?;
        tx.execute_batch(&format!(
            "CREATE TABLE {} AS SELECT * FROM read_parquet('{}');",
            self.config.name, msg.path
        ))?;

        let query = format!(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{}' AND column_name = '_key';",
            self.config.name
        );
        let mut stmt = tx.prepare(&query)?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        if count == 0 {
            tx.execute_batch(&format!(
                r"CREATE SEQUENCE keys_seq;
    ALTER TABLE {} ADD COLUMN _key UBIGINT DEFAULT NEXTVAL('keys_seq');",
                self.config.name,
            ))?;
        }
        tx.commit()?;
        Ok(())
    }
}

impl Handler<DbGetRowCount> for CollectionDbActor {
    type Result = Result<u64, ProjectError>;

    fn handle(&mut self, msg: DbGetRowCount, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let query = format!("SELECT COUNT('{}') FROM {};", msg.column, self.config.name);
        let mut stmt = self.conn.prepare(&query)?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        Ok(count as u64)
    }
}

impl Handler<DbCheckIndex> for CollectionDbActor {
    type Result = Result<bool, ProjectError>;

    fn handle(&mut self, msg: DbCheckIndex, _ctx: &mut SyncContext<Self>) -> Self::Result {
        Ok(self.vector_indices.contains_key(&msg.column))
    }
}

impl Handler<DbInitIndex> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbInitIndex, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let index_path = home_dir()
            .join("collections")
            .join(self.config.name.as_str())
            .join(self.config.index_dir.as_str())
            .join(&msg.column);

        let options = IndexOptions {
            dimensions: msg.dimensions,
            metric: MetricKind::Cos,
            quantization: msg.quantization,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: true,
        };

        let mut index = VectorIndex::new(index_path, true)?;
        index.with_options(&options, 20000)?;
        self.vector_indices.insert(msg.column, index);
        Ok(())
    }
}

impl Handler<DbGetBatch> for CollectionDbActor {
    type Result = Result<(Vec<String>, Vec<u64>), ProjectError>;

    fn handle(&mut self, msg: DbGetBatch, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {}, _key FROM {} LIMIT {} OFFSET {};",
            msg.column, self.config.name, msg.batch_size, msg.offset
        ))?;
        let result: Vec<RecordBatch> = stmt.query_arrow([])?.collect();
        if result.is_empty() {
            return Ok((vec![], vec![]));
        }
        let batch = &result[0];

        let col_array = batch.column_by_name(&msg.column).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let col_values: Vec<String> = col_array.iter().map(|s| s.unwrap().to_string()).collect();

        let key_array = batch.column_by_name("_key").unwrap().as_any().downcast_ref::<PrimitiveArray<UInt64Type>>().unwrap();
        let keys: Vec<u64> = key_array.iter().map(|key| key.unwrap_or(0)).collect();

        Ok((col_values, keys))
    }
}

impl Handler<DbAddEmbeddings> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbAddEmbeddings, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let index = self.vector_indices.get_mut(&msg.column).ok_or_else(|| {
            ProjectError::Anyhow(anyhow!("Vector index for column '{}' not found", msg.column))
        })?;

        match msg.embeddings {
            Embeddings::F16(emb) => {
                let (_, vector_dim) = emb.dim();
                index.add::<UsearchF16>(&msg.keys, emb.as_ptr() as *const UsearchF16, vector_dim)?;
            }
            Embeddings::F32(emb) => {
                let (_, vector_dim) = emb.dim();
                index.add::<f32>(&msg.keys, emb.as_ptr(), vector_dim)?;
            }
        }
        Ok(())
    }
}

impl Handler<DbSaveIndex> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbSaveIndex, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let index = self.vector_indices.get(&msg.column).ok_or_else(|| {
            ProjectError::Anyhow(anyhow!("Vector index for column '{}' not found", msg.column))
        })?;
        index.save()?;
        Ok(())
    }
}

impl Handler<DbSearchAndFetch> for CollectionDbActor {
    type Result = Result<Vec<SearchResult>, ProjectError>;

    fn handle(&mut self, msg: DbSearchAndFetch, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let index = self.vector_indices.get(&msg.column).ok_or_else(|| {
            ProjectError::Anyhow(anyhow!("Vector index for column '{}' not found", msg.column))
        })?;

        let similarity_results = match msg.query_embedding {
            Embeddings::F16(emb) => index.search::<UsearchF16>(
                emb.as_ptr() as *const UsearchF16,
                emb.dim().1,
                msg.limit,
            )?,
            Embeddings::F32(emb) => index.search::<f32>(
                emb.as_ptr(), 
                emb.dim().1, 
                msg.limit
            )?,
        };

        let keys: Vec<u64> = similarity_results.iter().map(|r| r.key).collect();
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let keys_str = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(", ");
        let query = format!(
            "SELECT _key, {} FROM {} WHERE _key IN ({});",
            msg.column, self.config.name, keys_str
        );
        let mut stmt = self.conn.prepare(&query)?;
        
        let rbs: Vec<RecordBatch> = stmt.query_arrow([])?.collect();
        let rb = rbs.first().ok_or_else(|| {
            ProjectError::Anyhow(anyhow!("No records found"))
        })?;
        
        let key_array = rb
            .column_by_name("_key")
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("Column '_key' not found")))?
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("_key is not of type UInt64")))?;
            
        let text_array = rb
            .column_by_name(&msg.column)
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("Column '{}' not found", msg.column)))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("Column is not of type String")))?;
            
        let mut content_map = key_array
            .iter()
            .zip(text_array.iter())
            .filter_map(|(k, v)| k.map(|k_val| (k_val, v.map(|v_val| v_val.to_string()))))
            .filter_map(|(k, v)| v.map(|v_val| (k, v_val)))
            .collect::<HashMap<_, _>>();
            
        let ordered_contents: Vec<String> = keys.iter().filter_map(|k| content_map.remove(k)).collect();

        let search_results = similarity_results
            .into_iter()
            .zip(ordered_contents.into_iter())
            .map(|(sim, content)| SearchResult {
                content,
                key: sim.key,
                score: sim.score,
            })
            .collect();

        Ok(search_results)
    }
}

// ---- CollectionActor ----

pub struct CollectionActor {
    config: CollectionConfig,
    model_manager: Addr<ModelManagerActor>,
    db_actor: Addr<CollectionDbActor>,
}

impl CollectionActor {
    pub fn new(config: CollectionConfig, model_manager: Addr<ModelManagerActor>) -> Self {
        let config_clone = config.clone();
        let db_actor = SyncArbiter::start(1, move || {
            CollectionDbActor::new(config_clone.clone())
        });

        Self {
            config,
            model_manager,
            db_actor,
        }
    }
}

impl Actor for CollectionActor {
    type Context = Context<Self>;
}

// ---- External Messages ----

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
        let db_actor = self.db_actor.clone();
        Box::pin(async move {
            db_actor.send(DbImportJsonl { path: msg.path }).await??;
            Ok(())
        })
    }
}

impl Handler<ImportParquet> for CollectionActor {
    type Result = ResponseFuture<Result<(), ProjectError>>;

    fn handle(&mut self, msg: ImportParquet, _ctx: &mut Context<Self>) -> Self::Result {
        let db_actor = self.db_actor.clone();
        Box::pin(async move {
            db_actor.send(DbImportParquet { path: msg.path }).await??;
            Ok(())
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
        let db_actor = self.db_actor.clone();
        let model_manager = self.model_manager.clone();

        Box::pin(async move {
            let column_name = msg.name;
            let batch_size = msg.batch_size;
            let model_id = msg.model_id;

            let count = db_actor.send(DbGetRowCount { column: column_name.clone() }).await??;
            let num_batches = (count + batch_size - 1) / batch_size;
            info!("Starting to index {} records from column '{}' in batches of {}", count, column_name, batch_size);

            let has_index = db_actor.send(DbCheckIndex { column: column_name.clone() }).await??;

            if !has_index {
                let (vector_dim, output_dtype) = model_manager
                    .send(GetModelMetadata { id: model_id })
                    .await??;

                let scalar_kind = match output_dtype {
                    ModelOutputDType::F32 => ScalarKind::F32,
                    ModelOutputDType::F16 => ScalarKind::F16,
                    ModelOutputDType::Int8 => ScalarKind::I8,
                };

                db_actor.send(DbInitIndex {
                    column: column_name.clone(),
                    dimensions: vector_dim as usize,
                    quantization: scalar_kind,
                }).await??;
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

                print!("\r{} / {} batches - ETA: {:?}", batch, total_steps, eta);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();

                let offset = batch * batch_size;

                let (texts, keys) = db_actor.send(DbGetBatch {
                    column: column_name.clone(),
                    batch_size,
                    offset,
                }).await??;

                if texts.is_empty() {
                    break;
                }

                let embeddings = model_manager
                    .send(Predict {
                        id: model_id,
                        texts,
                    })
                    .await??;

                db_actor.send(DbAddEmbeddings {
                    column: column_name.clone(),
                    keys,
                    embeddings,
                }).await??;
            }

            db_actor.send(DbSaveIndex { column: column_name.clone() }).await??;

            println!("");
            info!("Total duration: {:?}", start.elapsed());

            Ok(())
        })
    }
}

impl Handler<Search> for CollectionActor {
    type Result = ResponseFuture<Result<Vec<SearchResult>, ProjectError>>;

    fn handle(&mut self, msg: Search, _ctx: &mut Context<Self>) -> Self::Result {
        let db_actor = self.db_actor.clone();
        let model_manager = self.model_manager.clone();

        Box::pin(async move {
            let query_embedding = model_manager
                .send(Predict {
                    id: msg.model_id,
                    texts: vec![msg.query],
                })
                .await??;

            let search_results = db_actor.send(DbSearchAndFetch {
                column: msg.column,
                query_embedding,
                limit: msg.limit as usize,
            }).await??;

            Ok(search_results)
        })
    }
}