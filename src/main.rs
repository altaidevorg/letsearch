use actix::Actor;
use anyhow;
use chrono;
use clap::{Parser, Subcommand};
use env_logger::fmt::Formatter;
use indicatif::{ProgressBar, ProgressStyle};
use letsearch::actors::collection_actor::{
    AppendJsonl, AppendParquet, EmbedColumn, GetConfig, ImportJsonl, ImportParquet, ImportPdf,
    ImportTextFile, ImportWordFile,
};
use letsearch::actors::collection_manager_actor::{
    CollectionManagerActor, CreateCollection, GetModelIdForCollection, LoadCollection,
    SearchCollection,
};
use letsearch::actors::model_actor::{LoadModel, ModelManagerActor};
use letsearch::chunker::ChunkerConfig;
use letsearch::collection::collection_utils::{home_dir, CollectionConfig};
use letsearch::hf_ops::list_models;
use letsearch::serve::run_server;
use log::{info, Record};
use std::io::Write;
use std::time::Duration;

/// CLI application for indexing and searching documents
#[derive(Parser, Debug)]
#[command(
    name = "letsearch",
    version = "0.1.14",
    author = "yusufsarigoz@gmail.com",
    about = "Single binary to embed, index, serve and search your documents",
    subcommand_required = true,
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Index documents from JSONL, Parquet, PDF, Word, or plain text / Markdown files.
    Index {
        /// Path to file(s) to index.
        /// Supports `.jsonl`, `.parquet`, `.pdf`, `.doc`, `.docx`, `.txt`, `.md`, and `.markdown` (case-insensitive suffix).
        /// For JSONL/Parquet you can use local paths, `hf://datasets/...`, or glob patterns where DuckDB accepts them.
        #[arg(required = true)]
        files: String,

        /// name of the collection to be created
        #[arg(short, long, required = true)]
        collection_name: String,

        /// Model to create embeddings.
        /// You can also give a hf:// path and it will be automatically  downloaded.
        /// Use gemini://<model-name> (e.g. gemini://gemini-embedding-2-preview) to use
        /// a Gemini embedding model via the Google AI API.
        #[arg(short, long, default_value = "hf://mys/minilm")]
        model: String,

        /// model variant. f32, f16 and i8 are supported for now.
        #[arg(short, long, default_value = "f32")]
        variant: String,

        /// HuggingFace token. Only needed when you want to access private repos
        #[arg(long)]
        hf_token: Option<String>,

        /// Gemini API key. Required when using a gemini:// model.
        /// Falls back to the GEMINI_API_KEY environment variable when not provided.
        #[arg(long)]
        gemini_api_key: Option<String>,

        /// batch size when embedding texts
        #[arg(short, long, default_value = "32")]
        batch_size: u64,

        /// columns to embed and index for vector search.
        /// You can provide this option multiple times
        /// for multi-column indexing.
        #[arg(short, long, action = clap::ArgAction::Append)]
        index_columns: Vec<String>,

        /// remove and re-create collection if it exists
        #[arg(long, action=clap::ArgAction::SetTrue)]
        overwrite: bool,

        /// For PDF / Word / `.txt` / `.md` / `.markdown`: maximum tokens per chunk (after extraction).
        /// When omitted, the full document is stored as a single row (no splitting).
        #[arg(long)]
        chunk_max_tokens: Option<usize>,

        /// For PDF / Word / `.txt` / `.md` / `.markdown`: overlap tokens between consecutive chunks.
        #[arg(long, default_value = "50")]
        chunk_overlap_tokens: usize,

        /// For PDF / Word / `.txt` / `.md` / `.markdown`: path to a Hugging Face `tokenizer.json` for token counting.
        /// When omitted, a word-count approximation is used for chunking.
        #[arg(long)]
        tokenizer_path: Option<String>,
    },

    /// Create an empty collection: directory, `config.json`, and empty DuckDB only. No model load, no documents.
    /// Add data with `add-docs` or run `index` (see README for flows).
    Init {
        /// Collection name (directory under LETSEARCH_HOME/collections/)
        #[arg(short, long, required = true)]
        collection_name: String,

        /// Embedding model to record in config (used when you later run `add-docs` / embedding).
        #[arg(short, long, default_value = "hf://mys/minilm")]
        model: String,

        #[arg(short, long, default_value = "f32")]
        variant: String,

        /// Indexed column name(s). Repeat `--index-columns` for multiple. Defaults to `text` if omitted.
        #[arg(short, long, action = clap::ArgAction::Append)]
        index_columns: Vec<String>,

        /// Delete existing collection directory on disk if present, then recreate.
        #[arg(long, action = clap::ArgAction::SetTrue)]
        overwrite: bool,
    },

    /// serve a collection for search over web API
    Serve {
        /// collection to serve
        #[arg(short, long, required = true)]
        collection_name: String,

        /// host to listen to
        #[arg(short('H'), long, default_value = "127.0.0.1")]
        host: String,

        /// port to listen to
        #[arg(short, long, default_value = "7898")]
        port: i32,

        /// HuggingFace token. Only needed when you want to access private repos
        #[arg(long)]
        hf_token: Option<String>,

        /// Gemini API key. Required when the collection uses a gemini:// model.
        /// Falls back to the GEMINI_API_KEY environment variable when not provided.
        #[arg(long)]
        gemini_api_key: Option<String>,
    },

    /// list models compatible with letsearch
    ListModels {
        /// HuggingFace Token. Only required to access private models
        #[arg(long)]
        hf_token: Option<String>,
    },

    /// Search queries natively in the terminal
    Search {
        /// collection to search
        #[arg(short, long, required = true)]
        collection_name: String,

        /// Column to search (must match a name from `index --index-columns`, e.g. `txt` not `text`)
        #[arg(long, required = true)]
        column: String,

        /// Search query. In bash/zsh, wrap the whole phrase in single quotes if it contains spaces
        /// or ASCII double quotes (e.g. -q 'Balkan "savaşı'). Curly/smart quotes are not shell quotes.
        #[arg(short, long, required = true)]
        query: String,

        /// limit the number of search results
        #[arg(short, long, default_value = "10")]
        limit: u32,

        /// HuggingFace token. Only needed when you want to access private repos
        #[arg(long)]
        hf_token: Option<String>,

        /// Gemini API key. Required when the collection uses a gemini:// model.
        /// Falls back to the GEMINI_API_KEY environment variable when not provided.
        #[arg(long)]
        gemini_api_key: Option<String>,
    },

    /// Print raw rows from a collection (ordered by `_key`). No search query and no embedding model load.
    Sample {
        /// Collection name (same as `index --collection-name`)
        #[arg(short, long, required = true)]
        collection_name: String,

        /// Column to print (same name as `index --index-columns`, e.g. `text` or `txt`)
        #[arg(long, required = true)]
        column: String,

        /// Max rows to print
        #[arg(short, long, default_value = "10")]
        limit: u64,

        /// Skip this many rows (pagination)
        #[arg(long, default_value = "0")]
        offset: u64,
    },

    /// Add new documents to an existing collection for incremental indexing.
    /// Supports .jsonl, .parquet, .pdf, .doc, .docx, .txt, .md, and .markdown files.
    AddDocs {
        /// Path to the file to add.
        /// Supported formats: .jsonl, .parquet, .pdf, .doc, .docx, .txt, .md, .markdown
        #[arg(required = true)]
        files: String,

        /// Name of the existing collection to add documents to
        #[arg(short, long, required = true)]
        collection_name: String,

        /// batch size when embedding texts
        #[arg(short, long, default_value = "32")]
        batch_size: u64,

        /// For PDF / Word / `.txt` / `.md` / `.markdown`: target column for text chunks.
        /// Defaults to the first index column in the collection config, or "text".
        #[arg(long)]
        column: Option<String>,

        /// For PDF / Word / `.txt` / `.md` / `.markdown`: maximum tokens per chunk.
        #[arg(long)]
        chunk_max_tokens: Option<usize>,

        /// For PDF / Word / `.txt` / `.md` / `.markdown`: overlap tokens between consecutive chunks.
        #[arg(long, default_value = "50")]
        chunk_overlap_tokens: usize,

        /// For PDF / Word / `.txt` / `.md` / `.markdown`: path to tokenizer.json for accurate token counting.
        #[arg(long)]
        tokenizer_path: Option<String>,

        /// HuggingFace token. Only needed when you want to access private repos
        #[arg(long)]
        hf_token: Option<String>,

        /// Gemini API key. Required when the collection uses a gemini:// model.
        /// Falls back to the GEMINI_API_KEY environment variable when not provided.
        #[arg(long)]
        gemini_api_key: Option<String>,
    },
}

/// `.txt`, `.md`, `.markdown` — UTF-8 read + optional chunking (same pipeline as PDF text chunks).
fn is_plaintext_document_path(files_lower: &str) -> bool {
    files_lower.ends_with(".txt")
        || files_lower.ends_with(".md")
        || files_lower.ends_with(".markdown")
}

fn is_word_document_path(files_lower: &str) -> bool {
    files_lower.ends_with(".docx") || files_lower.ends_with(".doc")
}

/// Trims whitespace and strips Unicode “smart” quotes often pasted around paths in terminals.
fn sanitize_file_path_arg(s: &str) -> String {
    s.trim()
        .trim_matches(|c: char| {
            matches!(
                c,
                '\'' | '"' | '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}'
            )
        })
        .to_string()
}

fn is_safe_sql_identifier(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_')
}

fn duckdb_quote_ident_cli(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[actix::main]
async fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .format(|buf: &mut Formatter, record: &Record| {
            writeln!(
                buf,
                "[{} {}] {}",
                chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                record.level(),
                record.args()
            )
        })
        .filter_module("ort::execution_providers", log::LevelFilter::Error)
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse(); // Automatically parses the arguments into the struct

    match &cli.command {
        Commands::Index {
            files,
            collection_name,
            model,
            variant,
            hf_token,
            gemini_api_key,
            batch_size,
            index_columns,
            overwrite,
            chunk_max_tokens,
            chunk_overlap_tokens,
            tokenizer_path,
        } => {
            let mut config = CollectionConfig::default();
            config.name = collection_name.to_string();
            config.index_columns = index_columns.to_vec();
            config.model_name = model.to_string();
            config.model_variant = variant.to_string();

            let token = hf_token.clone().or_else(|| std::env::var("HF_TOKEN").ok());
            let gemini_key = gemini_api_key
                .clone()
                .or_else(|| std::env::var("GEMINI_API_KEY").ok());

            let model_manager_addr = ModelManagerActor::new().start();
            let collection_manager_addr = CollectionManagerActor::new(
                token.clone(),
                model_manager_addr.clone(),
                gemini_key.clone(),
            )
            .start();

            let collection_addr = collection_manager_addr
                .send(CreateCollection {
                    config,
                    overwrite: *overwrite,
                })
                .await??;
            info!("Collection '{}' created", collection_name);

            let file_path = sanitize_file_path_arg(files);
            let files_lower = file_path.to_ascii_lowercase();
            if files_lower.ends_with(".jsonl") {
                collection_addr
                    .send(ImportJsonl {
                        path: file_path.clone(),
                    })
                    .await??;
            } else if files_lower.ends_with(".parquet") {
                collection_addr
                    .send(ImportParquet {
                        path: file_path.clone(),
                    })
                    .await??;
            } else if files_lower.ends_with(".pdf") {
                if index_columns.len() != 1 {
                    return Err(anyhow::anyhow!(
                        "PDF indexing requires exactly one --index-columns <NAME> (VARCHAR column for chunked text)."
                    ));
                }
                let chunker_config = chunk_max_tokens.map(|max| ChunkerConfig {
                    max_tokens: max,
                    overlap_tokens: *chunk_overlap_tokens,
                    tokenizer_path: tokenizer_path.clone(),
                });
                collection_addr
                    .send(ImportPdf {
                        path: file_path.clone(),
                        column: index_columns[0].clone(),
                        chunker_config,
                    })
                    .await??;
                info!("Imported PDF into column '{}'", index_columns[0]);
            } else if is_word_document_path(&files_lower) {
                if index_columns.len() != 1 {
                    return Err(anyhow::anyhow!(
                        "Word (.doc / .docx) indexing requires exactly one --index-columns <NAME> (VARCHAR column for text chunks)."
                    ));
                }
                let chunker_config = chunk_max_tokens.map(|max| ChunkerConfig {
                    max_tokens: max,
                    overlap_tokens: *chunk_overlap_tokens,
                    tokenizer_path: tokenizer_path.clone(),
                });
                collection_addr
                    .send(ImportWordFile {
                        path: file_path.clone(),
                        column: index_columns[0].clone(),
                        chunker_config,
                    })
                    .await??;
                info!(
                    "Imported Word document into column '{}'",
                    index_columns[0]
                );
            } else if is_plaintext_document_path(&files_lower) {
                if index_columns.len() != 1 {
                    return Err(anyhow::anyhow!(
                        "Plain-text / Markdown indexing requires exactly one --index-columns <NAME> (VARCHAR column for text chunks)."
                    ));
                }
                let chunker_config = chunk_max_tokens.map(|max| ChunkerConfig {
                    max_tokens: max,
                    overlap_tokens: *chunk_overlap_tokens,
                    tokenizer_path: tokenizer_path.clone(),
                });
                collection_addr
                    .send(ImportTextFile {
                        path: file_path.clone(),
                        column: index_columns[0].clone(),
                        chunker_config,
                    })
                    .await??;
                info!(
                    "Imported text/Markdown file into column '{}'",
                    index_columns[0]
                );
            } else {
                return Err(anyhow::anyhow!(
                    "Unsupported file type. Use .jsonl, .parquet, .pdf, .doc, .docx, .txt, .md, or .markdown"
                ));
            }

            if !index_columns.is_empty() {
                let model_id = model_manager_addr
                    .send(LoadModel {
                        path: model.to_string(),
                        variant: variant.to_string(),
                        token,
                        gemini_api_key: gemini_key,
                    })
                    .await??;

                for column_name in index_columns {
                    collection_addr
                        .send(EmbedColumn {
                            name: column_name.to_string(),
                            batch_size: *batch_size,
                            model_id,
                        })
                        .await??;
                }
            }
        }

        Commands::Init {
            collection_name,
            model,
            variant,
            index_columns,
            overwrite,
        } => {
            let mut cols = index_columns.clone();
            if cols.is_empty() {
                cols.push("text".to_string());
            }
            for c in &cols {
                if !is_safe_sql_identifier(c) {
                    return Err(anyhow::anyhow!(
                        "Invalid --index-columns '{}': only letters, digits, and underscores are allowed",
                        c
                    ));
                }
            }

            let mut config = CollectionConfig::default();
            config.name = collection_name.to_string();
            config.index_columns = cols;
            config.model_name = model.to_string();
            config.model_variant = variant.to_string();

            let dir = CollectionConfig::init_on_disk(&config, *overwrite)?;
            info!(
                "Empty collection '{}' created at {} (configure columns: {:?}; add data with add-docs or index)",
                collection_name,
                dir.display(),
                config.index_columns
            );
        }

        Commands::Serve {
            collection_name,
            host,
            port,
            hf_token,
            gemini_api_key,
        } => {
            let token = hf_token.clone().or_else(|| std::env::var("HF_TOKEN").ok());
            let gemini_key = gemini_api_key
                .clone()
                .or_else(|| std::env::var("GEMINI_API_KEY").ok());

            run_server(
                host.to_string(),
                port.to_owned(),
                collection_name.to_string(),
                token,
                gemini_key,
            )
            .await?;
        }

        Commands::ListModels { hf_token } => {
            let token = hf_token.clone().or_else(|| std::env::var("HF_TOKEN").ok());
            list_models(token).await?;
        }

        Commands::Search {
            collection_name,
            column,
            query,
            limit,
            hf_token,
            gemini_api_key,
        } => {
            let token = hf_token.clone().or_else(|| std::env::var("HF_TOKEN").ok());
            let gemini_key = gemini_api_key
                .clone()
                .or_else(|| std::env::var("GEMINI_API_KEY").ok());

            let progress_bar = ProgressBar::new_spinner();
            progress_bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .expect("Failed to set template")
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            progress_bar.enable_steady_tick(Duration::from_millis(100));
            progress_bar.set_message("Loading models and collection into memory...");

            let model_manager_addr = ModelManagerActor::new().start();
            let collection_manager_addr =
                CollectionManagerActor::new(token.clone(), model_manager_addr.clone(), gemini_key)
                    .start();

            let load_result = collection_manager_addr
                .send(LoadCollection {
                    name: collection_name.to_string(),
                })
                .await;

            if let Err(e) = load_result
                .map_err(|e| anyhow::anyhow!(e))
                .and_then(|r| r.map_err(|e| anyhow::anyhow!(e)))
            {
                progress_bar.finish_and_clear();
                eprintln!("Failed to load collection '{}': {:?}", collection_name, e);
                std::process::exit(1);
            }

            progress_bar.set_message("Searching...");

            let search_result = collection_manager_addr
                .send(SearchCollection {
                    collection_name: collection_name.to_string(),
                    column: column.to_string(),
                    query: query.to_string(),
                    limit: *limit,
                })
                .await;

            progress_bar.finish_and_clear();

            match search_result {
                Ok(Ok(results)) => {
                    println!(
                        "\nFound {} result(s) for query: '{}'\n",
                        results.len(),
                        query
                    );
                    for (i, result) in results.iter().enumerate() {
                        println!("{}. [Score: {:.4}]", i + 1, result.score);
                        println!("---\n{}\n---", result.content);
                    }
                }
                Ok(Err(e)) => eprintln!("Search error: {:?}", e),
                Err(e) => eprintln!("Execution error: {:?}", e),
            }
        }

        Commands::Sample {
            collection_name,
            column,
            limit,
            offset,
        } => {
            if !is_safe_sql_identifier(column) {
                return Err(anyhow::anyhow!(
                    "Invalid column name: only letters, digits, and underscores are allowed"
                ));
            }

            let config = CollectionConfig::from_file(collection_name)?;
            let cap: u64 = 10_000;
            let limit = (*limit).min(cap);
            let offset = *offset;

            let db_path = home_dir()
                .join("collections")
                .join(config.name.as_str())
                .join(config.db_path.as_str());
            let conn = duckdb::Connection::open(&db_path).map_err(|e| {
                anyhow::anyhow!(
                    "Open DuckDB at {}: {} (same cwd / LETSEARCH_HOME as when you ran index?)",
                    db_path.display(),
                    e
                )
            })?;

            let table_sql = duckdb_quote_ident_cli(&config.name);
            let col_sql = duckdb_quote_ident_cli(column);
            let sql = format!(
                "SELECT _key, {col_sql} FROM {table_sql} ORDER BY _key LIMIT ? OFFSET ?"
            );

            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query(duckdb::params![limit as i64, offset as i64])?;

            println!(
                "Collection '{}' — up to {} row(s), offset {}\n",
                config.name, limit, offset
            );

            let mut n = 0usize;
            while let Some(row) = rows.next()? {
                let key: u64 = row.get(0)?;
                let text: Option<String> = row.get(1)?;
                n += 1;
                println!("--- _key: {} ---", key);
                println!("{}\n", text.unwrap_or_default());
            }
            if n == 0 {
                println!("(no rows in this range)");
            }
        }

        Commands::AddDocs {
            files,
            collection_name,
            batch_size,
            column,
            chunk_max_tokens,
            chunk_overlap_tokens,
            tokenizer_path,
            hf_token,
            gemini_api_key,
        } => {
            let token = hf_token.clone().or_else(|| std::env::var("HF_TOKEN").ok());
            let gemini_key = gemini_api_key
                .clone()
                .or_else(|| std::env::var("GEMINI_API_KEY").ok());

            let model_manager_addr = ModelManagerActor::new().start();
            let collection_manager_addr =
                CollectionManagerActor::new(token.clone(), model_manager_addr.clone(), gemini_key)
                    .start();

            let progress_bar = ProgressBar::new_spinner();
            progress_bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .expect("Failed to set template")
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            progress_bar.enable_steady_tick(Duration::from_millis(100));
            progress_bar.set_message(format!("Loading collection '{}'...", collection_name));

            let collection_addr = collection_manager_addr
                .send(LoadCollection {
                    name: collection_name.to_string(),
                })
                .await??;

            progress_bar.finish_and_clear();
            info!("Collection '{}' loaded", collection_name);

            // Fetch config once and reuse it throughout this command.
            let config = collection_addr.send(GetConfig).await??;

            // Import new data.
            let file_path = sanitize_file_path_arg(files);
            let files_lower = file_path.to_ascii_lowercase();
            if files_lower.ends_with(".jsonl") {
                collection_addr
                    .send(AppendJsonl {
                        path: file_path.clone(),
                    })
                    .await??;
                info!("Appended JSONL data from '{}'", file_path);
            } else if files_lower.ends_with(".parquet") {
                collection_addr
                    .send(AppendParquet {
                        path: file_path.clone(),
                    })
                    .await??;
                info!("Appended Parquet data from '{}'", file_path);
            } else if files_lower.ends_with(".pdf") {
                // Determine the target column.
                let target_col = column
                    .clone()
                    .or_else(|| config.index_columns.first().cloned())
                    .unwrap_or_else(|| "text".to_string());

                let chunker_config = chunk_max_tokens.map(|max| ChunkerConfig {
                    max_tokens: max,
                    overlap_tokens: *chunk_overlap_tokens,
                    tokenizer_path: tokenizer_path.clone(),
                });

                collection_addr
                    .send(ImportPdf {
                        path: file_path.clone(),
                        column: target_col,
                        chunker_config,
                    })
                    .await??;
                info!("Imported PDF from '{}'", file_path);
            } else if is_word_document_path(&files_lower) {
                let target_col = column
                    .clone()
                    .or_else(|| config.index_columns.first().cloned())
                    .unwrap_or_else(|| "text".to_string());

                let chunker_config = chunk_max_tokens.map(|max| ChunkerConfig {
                    max_tokens: max,
                    overlap_tokens: *chunk_overlap_tokens,
                    tokenizer_path: tokenizer_path.clone(),
                });

                collection_addr
                    .send(ImportWordFile {
                        path: file_path.clone(),
                        column: target_col.clone(),
                        chunker_config,
                    })
                    .await??;
                info!(
                    "Imported Word document from '{}' into column '{}'",
                    file_path, target_col
                );
            } else if is_plaintext_document_path(&files_lower) {
                let target_col = column
                    .clone()
                    .or_else(|| config.index_columns.first().cloned())
                    .unwrap_or_else(|| "text".to_string());

                let chunker_config = chunk_max_tokens.map(|max| ChunkerConfig {
                    max_tokens: max,
                    overlap_tokens: *chunk_overlap_tokens,
                    tokenizer_path: tokenizer_path.clone(),
                });

                collection_addr
                    .send(ImportTextFile {
                        path: file_path.clone(),
                        column: target_col.clone(),
                        chunker_config,
                    })
                    .await??;
                info!(
                    "Imported text/Markdown from '{}' into column '{}'",
                    file_path, target_col
                );
            } else {
                return Err(anyhow::anyhow!(
                    "Unsupported file format for add-docs: '{}' (use .jsonl, .parquet, .pdf, .doc, .docx, .txt, .md, .markdown)",
                    file_path
                ));
            }

            // Re-embed new rows for all configured index columns.
            if !config.index_columns.is_empty() {
                let model_id = collection_manager_addr
                    .send(GetModelIdForCollection {
                        name: collection_name.to_string(),
                    })
                    .await??;

                for column_name in &config.index_columns {
                    collection_addr
                        .send(EmbedColumn {
                            name: column_name.to_string(),
                            batch_size: *batch_size,
                            model_id,
                        })
                        .await??;
                }
            }
        }
    }

    Ok(())
}
