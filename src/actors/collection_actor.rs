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
use crate::chunker::ChunkerConfig;
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

/// Append rows from a JSONL file to an existing table.
#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbAppendJsonl {
    pub path: String,
}

/// Append rows from a Parquet file to an existing table.
#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbAppendParquet {
    pub path: String,
}

/// Insert a list of text chunks into the named column of the collection table.
/// Creates the table and/or column if they do not yet exist.
#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct DbImportMarkdownChunks {
    pub chunks: Vec<String>,
    pub column: String,
}

/// Return the number of vectors currently stored in the index for `column`.
/// Returns 0 when no index has been created yet.
#[derive(Message)]
#[rtype(result = "Result<u64, ProjectError>")]
pub struct DbGetIndexedCount {
    pub column: String,
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
        let collection_dir = home_dir().join("collections").join(config.name.as_str());

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

impl Handler<DbAppendJsonl> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbAppendJsonl, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let tx = self.conn.transaction()?;

        // Discover all columns except _key so the DEFAULT on _key is used.
        let cols_query = format!(
            "SELECT column_name FROM information_schema.columns \
             WHERE table_name = '{}' AND column_name != '_key' \
             ORDER BY ordinal_position;",
            self.config.name
        );
        let mut stmt = tx.prepare(&cols_query)?;
        let cols: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if cols.is_empty() {
            return Err(ProjectError::Anyhow(anyhow!(
                "Table '{}' has no columns to append to",
                self.config.name
            )));
        }
        let col_list = cols.join(", ");
        tx.execute_batch(&format!(
            "INSERT INTO {} ({}) SELECT {} FROM read_json_auto('{}');",
            self.config.name, col_list, col_list, msg.path
        ))?;
        tx.commit()?;
        Ok(())
    }
}

impl Handler<DbAppendParquet> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbAppendParquet, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let tx = self.conn.transaction()?;

        let cols_query = format!(
            "SELECT column_name FROM information_schema.columns \
             WHERE table_name = '{}' AND column_name != '_key' \
             ORDER BY ordinal_position;",
            self.config.name
        );
        let mut stmt = tx.prepare(&cols_query)?;
        let cols: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if cols.is_empty() {
            return Err(ProjectError::Anyhow(anyhow!(
                "Table '{}' has no columns to append to",
                self.config.name
            )));
        }
        let col_list = cols.join(", ");
        tx.execute_batch(&format!(
            "INSERT INTO {} ({}) SELECT {} FROM read_parquet('{}');",
            self.config.name, col_list, col_list, msg.path
        ))?;
        tx.commit()?;
        Ok(())
    }
}

impl Handler<DbImportMarkdownChunks> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(
        &mut self,
        msg: DbImportMarkdownChunks,
        _ctx: &mut SyncContext<Self>,
    ) -> Self::Result {
        if msg.chunks.is_empty() {
            return Ok(());
        }

        let tx = self.conn.transaction()?;

        // Check whether the table already exists.
        let table_exists: i64 = {
            let mut stmt = tx.prepare(&format!(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{}';",
                self.config.name
            ))?;
            stmt.query_row([], |row| row.get(0))?
        };

        if table_exists == 0 {
            // First import — create table with just the text column plus _key.
            tx.execute_batch(&format!(
                "CREATE TABLE {table} ({col} VARCHAR); \
                 CREATE SEQUENCE keys_seq; \
                 ALTER TABLE {table} ADD COLUMN _key UBIGINT DEFAULT NEXTVAL('keys_seq');",
                table = self.config.name,
                col = msg.column,
            ))?;
        } else {
            // Table exists — ensure the target column is present.
            let col_exists: i64 = {
                let mut stmt = tx.prepare(&format!(
                    "SELECT COUNT(*) FROM information_schema.columns \
                     WHERE table_name = '{}' AND column_name = '{}';",
                    self.config.name, msg.column
                ))?;
                stmt.query_row([], |row| row.get(0))?
            };
            if col_exists == 0 {
                tx.execute_batch(&format!(
                    "ALTER TABLE {} ADD COLUMN {} VARCHAR;",
                    self.config.name, msg.column
                ))?;
            }
        }

        // Insert each chunk using a parameterised statement.
        let insert_sql = format!(
            "INSERT INTO {} ({}) VALUES (?);",
            self.config.name, msg.column
        );
        let mut stmt = tx.prepare(&insert_sql)?;
        for chunk in &msg.chunks {
            stmt.execute(duckdb::params![chunk.as_str()])?;
        }

        tx.commit()?;
        Ok(())
    }
}

impl Handler<DbGetIndexedCount> for CollectionDbActor {
    type Result = Result<u64, ProjectError>;

    fn handle(&mut self, msg: DbGetIndexedCount, _ctx: &mut SyncContext<Self>) -> Self::Result {
        if let Some(index) = self.vector_indices.get(&msg.column) {
            if let Some(idx) = &index.index {
                return Ok(idx.size() as u64);
            }
        }
        Ok(0)
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

        let col_array = batch
            .column_by_name(&msg.column)
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("Column '{}' not found", msg.column)))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("Column is not of type String")))?;
        let col_values: Vec<String> = col_array
            .iter()
            .map(|s| s.unwrap_or_default().to_string())
            .collect();

        let key_array = batch
            .column_by_name("_key")
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("_key column not found")))?
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("_key is not of type UInt64")))?;
        let keys: Vec<u64> = key_array.iter().map(|key| key.unwrap_or(0)).collect();

        Ok((col_values, keys))
    }
}

impl Handler<DbAddEmbeddings> for CollectionDbActor {
    type Result = Result<(), ProjectError>;

    fn handle(&mut self, msg: DbAddEmbeddings, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let index = self.vector_indices.get_mut(&msg.column).ok_or_else(|| {
            ProjectError::Anyhow(anyhow!(
                "Vector index for column '{}' not found",
                msg.column
            ))
        })?;

        match msg.embeddings {
            Embeddings::F16(emb) => {
                let (_, vector_dim) = emb.dim();
                index.add::<UsearchF16>(
                    &msg.keys,
                    emb.as_ptr() as *const UsearchF16,
                    vector_dim,
                )?;
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
            ProjectError::Anyhow(anyhow!(
                "Vector index for column '{}' not found",
                msg.column
            ))
        })?;
        index.save()?;
        Ok(())
    }
}

impl Handler<DbSearchAndFetch> for CollectionDbActor {
    type Result = Result<Vec<SearchResult>, ProjectError>;

    fn handle(&mut self, msg: DbSearchAndFetch, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let index = self.vector_indices.get(&msg.column).ok_or_else(|| {
            ProjectError::Anyhow(anyhow!(
                "Vector index for column '{}' not found",
                msg.column
            ))
        })?;

        let similarity_results = match msg.query_embedding {
            Embeddings::F16(emb) => index.search::<UsearchF16>(
                emb.as_ptr() as *const UsearchF16,
                emb.dim().1,
                msg.limit,
            )?,
            Embeddings::F32(emb) => index.search::<f32>(emb.as_ptr(), emb.dim().1, msg.limit)?,
        };

        let keys: Vec<u64> = similarity_results.iter().map(|r| r.key).collect();
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let keys_str = keys
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!(
            "SELECT _key, {} FROM {} WHERE _key IN ({});",
            msg.column, self.config.name, keys_str
        );
        let mut stmt = self.conn.prepare(&query)?;

        let rbs: Vec<RecordBatch> = stmt.query_arrow([])?.collect();
        let rb = rbs
            .first()
            .ok_or_else(|| ProjectError::Anyhow(anyhow!("No records found")))?;

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

        let ordered_contents: Vec<String> =
            keys.iter().filter_map(|k| content_map.remove(k)).collect();

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
        let db_actor = SyncArbiter::start(1, move || CollectionDbActor::new(config_clone.clone()));

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

/// Append rows from a JSONL file to an existing collection table.
#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct AppendJsonl {
    pub path: String,
}

/// Append rows from a Parquet file to an existing collection table.
#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct AppendParquet {
    pub path: String,
}

/// Import a PDF document: convert to Markdown, optionally chunk it, and
/// insert the resulting chunks into the named column of the collection table.
#[derive(Message)]
#[rtype(result = "Result<(), ProjectError>")]
pub struct ImportPdf {
    /// Path to the PDF file.
    pub path: String,
    /// Target column name (e.g. `"text"`).
    pub column: String,
    /// Optional chunker configuration.  When `None` the full Markdown string
    /// is inserted as a single row.
    pub chunker_config: Option<ChunkerConfig>,
}

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

impl Handler<AppendJsonl> for CollectionActor {
    type Result = ResponseFuture<Result<(), ProjectError>>;

    fn handle(&mut self, msg: AppendJsonl, _ctx: &mut Context<Self>) -> Self::Result {
        let db_actor = self.db_actor.clone();
        Box::pin(async move {
            db_actor.send(DbAppendJsonl { path: msg.path }).await??;
            Ok(())
        })
    }
}

impl Handler<AppendParquet> for CollectionActor {
    type Result = ResponseFuture<Result<(), ProjectError>>;

    fn handle(&mut self, msg: AppendParquet, _ctx: &mut Context<Self>) -> Self::Result {
        let db_actor = self.db_actor.clone();
        Box::pin(async move {
            db_actor.send(DbAppendParquet { path: msg.path }).await??;
            Ok(())
        })
    }
}

impl Handler<ImportPdf> for CollectionActor {
    type Result = ResponseFuture<Result<(), ProjectError>>;

    fn handle(&mut self, msg: ImportPdf, _ctx: &mut Context<Self>) -> Self::Result {
        let db_actor = self.db_actor.clone();

        Box::pin(async move {
            let path = msg.path.clone();
            let column = msg.column.clone();
            let cfg = msg.chunker_config.clone();

            // PDF conversion is CPU/IO-bound — run it on a blocking thread.
            let chunks: Vec<String> = tokio::task::spawn_blocking(move || {
                let markdown = crate::pdf::pdf_to_markdown(&path)?;
                if let Some(chunker_cfg) = cfg {
                    let chunker = crate::chunker::MarkdownChunker::new(chunker_cfg)?;
                    anyhow::Ok(chunker.chunk(&markdown))
                } else {
                    anyhow::Ok(vec![markdown])
                }
            })
            .await
            .map_err(ProjectError::JoinError)??;

            db_actor
                .send(DbImportMarkdownChunks { chunks, column })
                .await??;
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

            let count = db_actor
                .send(DbGetRowCount {
                    column: column_name.clone(),
                })
                .await??;

            let has_index = db_actor
                .send(DbCheckIndex {
                    column: column_name.clone(),
                })
                .await??;

            if !has_index {
                let (vector_dim, output_dtype) = model_manager
                    .send(GetModelMetadata { id: model_id })
                    .await??;

                let scalar_kind = match output_dtype {
                    ModelOutputDType::F32 => ScalarKind::F32,
                    ModelOutputDType::F16 => ScalarKind::F16,
                    ModelOutputDType::Int8 => ScalarKind::I8,
                };

                db_actor
                    .send(DbInitIndex {
                        column: column_name.clone(),
                        dimensions: vector_dim as usize,
                        quantization: scalar_kind,
                    })
                    .await??;
            }

            // For incremental indexing: skip rows that are already indexed.
            let already_indexed = db_actor
                .send(DbGetIndexedCount {
                    column: column_name.clone(),
                })
                .await??;
            let start_offset = already_indexed;
            let remaining = count.saturating_sub(start_offset);
            let num_batches = (remaining + batch_size - 1) / batch_size;

            info!(
                "Starting to index {} new records from column '{}' in batches of {} (skipping {} already indexed)",
                remaining, column_name, batch_size, start_offset
            );

            if remaining == 0 {
                info!("Column '{}' is already fully indexed", column_name);
                return Ok(());
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
                let _ = std::io::Write::flush(&mut std::io::stdout());

                let offset = start_offset + batch * batch_size;

                let (texts, keys) = db_actor
                    .send(DbGetBatch {
                        column: column_name.clone(),
                        batch_size,
                        offset,
                    })
                    .await??;

                if texts.is_empty() {
                    break;
                }

                let embeddings = model_manager
                    .send(Predict {
                        id: model_id,
                        texts,
                    })
                    .await??;

                db_actor
                    .send(DbAddEmbeddings {
                        column: column_name.clone(),
                        keys,
                        embeddings,
                    })
                    .await??;
            }

            db_actor
                .send(DbSaveIndex {
                    column: column_name.clone(),
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
        let db_actor = self.db_actor.clone();
        let model_manager = self.model_manager.clone();

        Box::pin(async move {
            let query_embedding = model_manager
                .send(Predict {
                    id: msg.model_id,
                    texts: vec![msg.query],
                })
                .await??;

            let search_results = db_actor
                .send(DbSearchAndFetch {
                    column: msg.column,
                    query_embedding,
                    limit: msg.limit as usize,
                })
                .await??;

            Ok(search_results)
        })
    }
}
