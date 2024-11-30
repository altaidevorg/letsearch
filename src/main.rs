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
    name = "letsearche",
    version = "0.1.0",
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
        /// Path to files to index
        #[arg(required = true, num_args(1..), action = clap::ArgAction::Append)]
        files: Vec<String>,

        /// name of the collection to be created
        #[arg(short, long, required = true)]
        collection_name: String,

        /// Model to create embeddings
        #[arg(short, long, default_value = "minilm")]
        model: String,

        /// batch size when embedding texts
        #[arg(short, long, default_value = "32")]
        batch_size: u64,

        /// columns to embed and index for vector search
        #[arg(short, long, action = clap::ArgAction::Append)]
        index_columns: Vec<String>,

        /// remove and re-create collection directory if it exists
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
            batch_size,
            index_columns,
            overwrite,
        } => {
            let mut config = CollectionConfig::default();
            config.name = collection_name.to_string();
            config.index_columns = index_columns.to_vec();
            config.model_name = model.to_string();
            let collection_manager = CollectionManager::new();
            collection_manager
                .create_collection(config, overwrite.to_owned())
                .await?;
            info!("Collection '{}' created", collection_name);
            let jsonl_path = &files[0];
            collection_manager
                .import_jsonl(&collection_name, jsonl_path)
                .await?;

            if index_columns.len() > 0 {
                collection_manager
                    .embed_column(&collection_name, &index_columns[0], batch_size.to_owned())
                    .await
                    .unwrap();
            }
        }

        Commands::Serve {
            collection_name,
            host,
            port,
        } => {
            run_server(
                host.to_string(),
                port.to_owned(),
                collection_name.to_string(),
            )
            .await
            .unwrap();
        }
    }

    Ok(())
}

mod collection;
mod model;
mod serve;
