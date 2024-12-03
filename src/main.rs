use crate::collection::collection_utils::CollectionConfig;
use crate::serve::run_server;
use anyhow;
use chrono;
use clap::{Parser, Subcommand};
use collection::collection_manager::CollectionManager;
use env_logger::fmt::Formatter;
use log::{info, Record};
use std::io::Write;

/// CLI application for indexing and searching documents
#[derive(Parser, Debug)]
#[command(
    name = "letsearch",
    version = "0.1.9,
    author = "yusufsarigoz@gmail.com",
    about = "Index and search your documents, and serve it if you wish",
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
        #[arg(short, long, required = true)]
        model: String,

        /// model variant. f32, f16 and i8 are supported for now.
        #[arg(short, long, default_value = "f32")]
        variant: String,

        /// HuggingFace token. Only needed when you want to access private repos
        #[arg(long)]
        hf_token: Option<String>,

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
    },
}

#[tokio::main]
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
            batch_size,
            index_columns,
            overwrite,
        } => {
            let mut config = CollectionConfig::default();
            config.name = collection_name.to_string();
            config.index_columns = index_columns.to_vec();
            config.model_name = model.to_string();
            config.model_variant = variant.to_string();
            let collection_manager = CollectionManager::new(hf_token.to_owned());
            collection_manager
                .create_collection(config, overwrite.to_owned())
                .await?;
            info!("Collection '{}' created", collection_name);

            if files.ends_with(".jsonl") {
                collection_manager
                    .import_jsonl(&collection_name, files)
                    .await?;
            } else if files.ends_with(".parquet") {
                collection_manager
                    .import_parquet(&collection_name, files)
                    .await?;
            }

            if !index_columns.is_empty() {
                for column_name in index_columns {
                    collection_manager
                        .embed_column(&collection_name, column_name, batch_size.to_owned())
                        .await?;
                }
            }
        }

        Commands::Serve {
            collection_name,
            host,
            port,
            hf_token,
        } => {
            run_server(
                host.to_string(),
                port.to_owned(),
                collection_name.to_string(),
                hf_token.to_owned(),
            )
            .await?;
        }
    }

    Ok(())
}

mod collection;
mod hf_ops;
mod model;
mod serve;
