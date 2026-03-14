use crate::actors::model_actor::{ModelManagerActor, Predict, GetModelMetadata};
use crate::collection::collection_utils::{home_dir, CollectionConfig, SearchResult};
use crate::collection::vector_index::VectorIndex;
use crate::error::ProjectError;
use crate::model::model_utils::{Embeddings, ModelOutputDType};
use actix::prelude::*;
use anyhow::anyhow;
use duckdb::arrow::array::{PrimitiveArray, StringArray};
use duckdb::arrow::datatypes::UInt64Type;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::Connection;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use log::{info, debug};
use usearch::f16 as UsearchF16;
use usearch::{IndexOptions, MetricKind, ScalarKind};

pub struct SharedConnection(pub duckdb::Connection);
unsafe impl Send for SharedConnection {}
unsafe impl Sync for SharedConnection {}

// ---- Actor Definition ----
pub struct CollectionActor {
    config: CollectionConfig,
    db_path: PathBuf,
    vector_indices: Arc<RwLock<HashMap<String, Arc<RwLock<VectorIndex>>>>>,
    model_manager: Addr<ModelManagerActor>,
    conn: Arc<Mutex<SharedConnection>>,
}

impl CollectionActor {
    pub fn new(config: CollectionConfig, model_manager: Addr<ModelManagerActor>) -> Self {
        let collection_dir = home_dir()
            .join("collections")
            .join(config.name.as_str());
            
        // ensure dir exists
        std::fs::create_dir_all(&collection_dir).unwrap();
        
        let db_path = collection_dir.join(config.db_path.as_str());
        let conn = Connection::open(&db_path).expect("Failed to open DuckDB connection");
            
        let vector_indices = RwLock::new(HashMap::new());
        let index_dir = collection_dir.join(config.index_dir.as_str());
        if index_dir.exists() && !config.index_columns.is_empty() {
            let mut indices_guard = vector_indices.write().unwrap();
            for index_column in config.index_columns.clone() {
                let index_path = index_dir.join(index_column.as_str());
                if let Ok(vector_index) = VectorIndex::from(index_path.to_path_buf()) {
                    indices_guard.insert(index_column.clone(), Arc::new(RwLock::new(vector_index)));
                }
            }
        }
        
        Self {
            config,
            db_path,
            vector_indices: Arc::new(vector_indices),
            model_manager,
            conn: Arc::new(Mutex::new(SharedConnection(conn))),
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
        let collection_name = self.config.name.clone();
        let conn_arc = self.conn.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || -> Result<(), ProjectError> {
                let mut conn_guard = conn_arc.lock().unwrap();
                let tx = conn_guard.0.transaction()?;
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
        let collection_name = self.config.name.clone();
        let conn_arc = self.conn.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || -> Result<(), ProjectError> {
                let mut conn_guard = conn_arc.lock().unwrap();
                let tx = conn_guard.0.transaction()?;
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

    fn handle(&mut self, msg: EmbedColumn, _ctx: &mut Context<Self>) -> Self::Result {
        let collection_name = self.config.name.clone();
        let index_dir = self.config.index_dir.clone();
        let conn_arc = self.conn.clone();
        let vector_indices_arc = self.vector_indices.clone();
        let model_manager = self.model_manager.clone();

        Box::pin(async move {
            let column_name = msg.name;
            let batch_size = msg.batch_size;
            let model_id = msg.model_id;

            // 1. Get row count
            let count: u64 = tokio::task::spawn_blocking({
                let conn_arc = conn_arc.clone();
                let column_name = column_name.clone();
                let collection_name = collection_name.clone();
                move || -> Result<u64, ProjectError> {
                    let conn_guard = conn_arc.lock().unwrap();
                    let query = format!("SELECT COUNT('{}') FROM {};", column_name, collection_name);
                    let mut stmt = conn_guard.0.prepare(&query)?;
                    let count: i64 = stmt.query_row([], |row| row.get(0))?;
                    Ok(count as u64)
                }
            })
            .await??;

            let num_batches = (count + batch_size - 1) / batch_size;
            info!("Starting to index {} records from column '{}' in batches of {}", count, column_name, batch_size);

            // 2. Initialize index if it doesn't exist
            let needs_index_init = {
                let indices_guard = vector_indices_arc.read().unwrap();
                !indices_guard.contains_key(&column_name)
            };

            if needs_index_init {
                let (vector_dim, output_dtype) = model_manager
                    .send(GetModelMetadata { id: model_id })
                    .await??;

                let scalar_kind = match output_dtype {
                    ModelOutputDType::F32 => ScalarKind::F32,
                    ModelOutputDType::F16 => ScalarKind::F16,
                    ModelOutputDType::Int8 => ScalarKind::I8,
                };

                let index_path = home_dir()
                    .join("collections")
                    .join(collection_name.as_str())
                    .join(index_dir.as_str())
                    .join(&column_name);

                let options = IndexOptions {
                    dimensions: vector_dim as usize,
                    metric: MetricKind::Cos,
                    quantization: scalar_kind,
                    connectivity: 0,
                    expansion_add: 0,
                    expansion_search: 0,
                    multi: true,
                };

                let mut index = VectorIndex::new(index_path, true)?;
                index.with_options(&options, 20000)?;
                
                let mut indices_guard = vector_indices_arc.write().unwrap();
                indices_guard.insert(column_name.clone(), Arc::new(RwLock::new(index)));
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

                // 3. Extract batch of texts and keys
                let (texts, keys) = tokio::task::spawn_blocking({
                    let conn_arc = conn_arc.clone();
                    let column_name = column_name.clone();
                    let collection_name = collection_name.clone();
                    move || -> Result<(Vec<String>, Vec<u64>), ProjectError> {
                        let conn_guard = conn_arc.lock().unwrap();
                        let mut stmt = conn_guard.0.prepare(&format!(
                            "SELECT {}, _key FROM {} LIMIT {} OFFSET {};",
                            column_name, collection_name, batch_size, offset
                        ))?;
                        let result: Vec<RecordBatch> = stmt.query_arrow([])?.collect();
                        if result.is_empty() {
                            return Ok((vec![], vec![]));
                        }
                        let batch = &result[0];

                        let col_array = batch.column_by_name(&column_name).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
                        let col_values: Vec<String> = col_array.iter().map(|s| s.unwrap().to_string()).collect();

                        let key_array = batch.column_by_name("_key").unwrap().as_any().downcast_ref::<PrimitiveArray<UInt64Type>>().unwrap();
                        let keys: Vec<u64> = key_array.iter().map(|key| key.unwrap_or(0)).collect();

                        Ok((col_values, keys))
                    }
                })
                .await??;

                if texts.is_empty() {
                    break;
                }

                // 4. Generate Embeddings
                let embeddings = model_manager
                    .send(Predict {
                        id: model_id,
                        texts: texts.clone(),
                    })
                    .await??;

                // 5. Add to Usearch Index
                tokio::task::spawn_blocking({
                    let vector_indices_arc = vector_indices_arc.clone();
                    let column_name = column_name.clone();
                    move || -> Result<(), ProjectError> {
                        let indices_guard = vector_indices_arc.read().unwrap();
                        let index_arc = indices_guard.get(&column_name).unwrap().clone();
                        let mut index_guard = index_arc.write().unwrap();

                        match embeddings {
                            Embeddings::F16(emb) => {
                                let (_, vector_dim) = emb.dim();
                                index_guard.add::<UsearchF16>(&keys, emb.as_ptr() as *const UsearchF16, vector_dim)?;
                            }
                            Embeddings::F32(emb) => {
                                let (_, vector_dim) = emb.dim();
                                index_guard.add::<f32>(&keys, emb.as_ptr(), vector_dim)?;
                            }
                        }
                        Ok(())
                    }
                })
                .await??;
            }

            // 6. Save index to disk
            tokio::task::spawn_blocking({
                let vector_indices_arc = vector_indices_arc.clone();
                let column_name = column_name.clone();
                move || -> Result<(), ProjectError> {
                    let indices_guard = vector_indices_arc.read().unwrap();
                    let index_arc = indices_guard.get(&column_name).unwrap().clone();
                    let index_guard = index_arc.read().unwrap();
                    index_guard.save()?;
                    Ok(())
                }
            })
            .await??;

            println!("");
            info!("Total duration: {:?}", start.elapsed());

            Ok(())
        })
    }
}

impl Handler<Search> for CollectionActor {
    type Result = ResponseFuture<Result<Vec<SearchResult>, ProjectError>>;

    fn handle(&mut self, msg: Search, _ctx: &mut Context<Self>) -> Self::Result {
        let vector_indices_arc = self.vector_indices.clone();
        let model_manager = self.model_manager.clone();
        let conn_arc = self.conn.clone();
        let collection_name = self.config.name.clone();

        Box::pin(async move {
            let index_arc = {
                let indices_guard = vector_indices_arc.read().unwrap();
                indices_guard.get(&msg.column).cloned()
            };

            let index_arc = index_arc.ok_or_else(|| {
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
                let index_arc = index_arc.clone();
                move || -> Result<_, ProjectError> {
                    let index_guard = index_arc.read().unwrap();
                    let results = match query_embedding {
                        Embeddings::F16(emb) => index_guard.search::<UsearchF16>(
                            emb.as_ptr() as *const UsearchF16,
                            emb.dim().1,
                            msg.limit as usize,
                        )?,
                        Embeddings::F32(emb) => index_guard.search::<f32>(
                            emb.as_ptr(), 
                            emb.dim().1, 
                            msg.limit as usize
                        )?,
                    };
                    Ok(results)
                }
            })
            .await??;

            let keys: Vec<u64> = similarity_results.iter().map(|r| r.key).collect();
            if keys.is_empty() {
                return Ok(Vec::new());
            }

            let contents: Vec<String> = tokio::task::spawn_blocking({
                let conn_arc = conn_arc.clone();
                let collection_name = collection_name.clone();
                let column_name = column_name.clone();
                move || -> Result<_, ProjectError> {
                    let conn_guard = conn_arc.lock().unwrap();
                    let keys_str = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(", ");
                    let query = format!(
                        "SELECT _key, {} FROM {} WHERE _key IN ({});",
                        column_name, collection_name, keys_str
                    );
                    let mut stmt = conn_guard.0.prepare(&query)?;
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
                        .column_by_name(&column_name)
                        .ok_or_else(|| ProjectError::Anyhow(anyhow!("Column '{}' not found", column_name)))?
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| ProjectError::Anyhow(anyhow!("Column is not of type String")))?;
                        
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