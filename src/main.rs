mod collection;
mod model;
mod serve;
use crate::collection::Collection;
use crate::model::manager::ModelManager;
use crate::model::traits::Backend;
use crate::serve::run_server;
use anyhow;
use clap::{Parser, Subcommand};
use env_logger;
use log::{debug, info};
use std::time::Instant;

/// CLI application for indexing and searching documents
#[derive(Parser, Debug)]
#[command(
    name = "searche",
    version = "0.1.0",
    author = "yusufsarigoz@gmail.com",
    about = "Index and search your documents, and serve it if you wish",
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Index documents
    Index {
        /// Path to files to index
        #[arg(required = true, num_args(1..), action = clap::ArgAction::Append)]
        files: Vec<String>,

        /// name of the collection to be created
        #[arg(short, long, required = true)]
        collection_name: String,

        /// Model to create embeddings
        #[arg(short, long, default_value = "bge-m3")]
        model: String,

        /// Enable verbose output
        #[arg(short, long, action = clap::ArgAction::SetTrue)]
        verbose: bool,

        /// remove and re-create collection directory if it exists
        #[arg(long, action=clap::ArgAction::SetTrue)]
        overwrite: bool,
    },

    /// serve a collection for search over web API
    Serve {
        /// host to listen to
        #[arg(short, long, default_value = "127.0.0.1")]
        host: String,

        /// port to listen to
        #[arg(short, long, default_value = "7898")]
        port: i32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_module("ort::execution_providers", log::LevelFilter::Error)
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse(); // Automatically parses the arguments into the struct

    match &cli.command {
        Commands::Index {
            files,
            collection_name,
            model,
            verbose,
            overwrite,
        } => {
            let collection =
                Collection::new(collection_name.to_string(), overwrite.to_owned()).unwrap();
            let jsonl_path = &files[0];
            collection.import_jsonl(jsonl_path)?;
            let model_manager = ModelManager::new();
            let model_id = model_manager
                .load_model(model.to_string(), Backend::ONNX)
                .await
                .unwrap();
            info!("model loaded successfully");
            let start = Instant::now();
            let res = model_manager
                .predict(model_id, "this is a test")
                .await
                .unwrap();
            info!("it took: {:?}", start.elapsed());
            debug!("{res}");
        }

        Commands::Serve { host, port } => {
            run_server(host.to_string(), port.to_owned()).await.unwrap();
        }
    }

    Ok(())
}
