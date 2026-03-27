use actix::Actor;
use anyhow;
use chrono;
use clap::{Parser, Subcommand};
use env_logger::fmt::Formatter;
use indicatif::{ProgressBar, ProgressStyle};
use letsearch::actors::collection_actor::{
    AppendJsonl, AppendParquet, EmbedColumn, GetConfig, ImportJsonl, ImportParquet, ImportPdf,
};
use letsearch::actors::collection_manager_actor::{
    CollectionManagerActor, CreateCollection, GetModelIdForCollection, LoadCollection,
    SearchCollection,
};
use letsearch::actors::model_actor::{LoadModel, ModelManagerActor};
use letsearch::chunker::ChunkerConfig;
use letsearch::collection::collection_utils::CollectionConfig;
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
    /// Index documents
    Index {
        /// Path to file(s) to index.
        /// You can provide local or hf://datasets paths.
        /// It might be  a regular  path (absolute
        /// or relative), or a glob pattern.
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

        /// target column to search against
        #[arg(long, required = true)]
        column: String,

        /// your search query
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

    /// Add new documents to an existing collection for incremental indexing.
    /// Supports .jsonl, .parquet, and .pdf files.
    AddDocs {
        /// Path to the file to add.
        /// Supported formats: .jsonl, .parquet, .pdf
        #[arg(required = true)]
        files: String,

        /// Name of the existing collection to add documents to
        #[arg(short, long, required = true)]
        collection_name: String,

        /// batch size when embedding texts
        #[arg(short, long, default_value = "32")]
        batch_size: u64,

        /// For PDF files: target column name to store extracted text chunks.
        /// Defaults to the first index column in the collection config, or "text".
        #[arg(long)]
        column: Option<String>,

        /// For PDF files: maximum number of tokens per chunk.
        #[arg(long)]
        chunk_max_tokens: Option<usize>,

        /// For PDF files: number of overlap tokens between consecutive chunks.
        #[arg(long, default_value = "50")]
        chunk_overlap_tokens: usize,

        /// For PDF files: path to a tokenizer.json file for accurate token counting.
        /// When not provided, a word-count approximation is used.
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

            if files.ends_with(".jsonl") {
                collection_addr
                    .send(ImportJsonl {
                        path: files.to_string(),
                    })
                    .await??;
            } else if files.ends_with(".parquet") {
                collection_addr
                    .send(ImportParquet {
                        path: files.to_string(),
                    })
                    .await??;
            } else {
                return Err(anyhow::anyhow!("This file is currently not supported"));
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
            if files.ends_with(".jsonl") {
                collection_addr
                    .send(AppendJsonl {
                        path: files.to_string(),
                    })
                    .await??;
                info!("Appended JSONL data from '{}'", files);
            } else if files.ends_with(".parquet") {
                collection_addr
                    .send(AppendParquet {
                        path: files.to_string(),
                    })
                    .await??;
                info!("Appended Parquet data from '{}'", files);
            } else if files.ends_with(".pdf") {
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
                        path: files.to_string(),
                        column: target_col,
                        chunker_config,
                    })
                    .await??;
                info!("Imported PDF from '{}'", files);
            } else {
                return Err(anyhow::anyhow!(
                    "Unsupported file format for add-docs: '{}'",
                    files
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
