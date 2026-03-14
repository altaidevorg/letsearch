extern crate letsearch;
use criterion::{criterion_group, criterion_main, Criterion};

use actix::prelude::*;
use letsearch::actors::collection_actor::ImportJsonl;
#[cfg(feature = "heavyweight")]
use letsearch::actors::collection_actor::{EmbedColumn, ImportParquet};
use letsearch::actors::collection_manager_actor::{CollectionManagerActor, CreateCollection};
#[cfg(feature = "heavyweight")]
use letsearch::actors::collection_manager_actor::GetModelIdForCollection;
use letsearch::actors::model_actor::ModelManagerActor;
use letsearch::collection::collection_utils::CollectionConfig;


pub async fn import_jsonl(files: &str, collection_name: &str) -> anyhow::Result<()> {
    let mut config = CollectionConfig::default();
    config.name = collection_name.to_string();

    let model_manager = ModelManagerActor::new().start();
    let collection_manager = CollectionManagerActor::new(None, model_manager).start();

    let collection_addr = collection_manager
        .send(CreateCollection {
            config,
            overwrite: true,
        })
        .await??;

    collection_addr
        .send(ImportJsonl {
            path: files.to_string(),
        })
        .await??;
    Ok(())
}

fn benchmark_import_jsonl(c: &mut Criterion) {
    let runner = tokio::runtime::Runtime::new().unwrap();
    c.bench_function("import_jsonl", |b| {
        b.to_async(&runner).iter(|| async {
            let files = "./benches/rag_instruct_benchmark_tester.jsonl";
            let collection_name = "test_collection";

            import_jsonl(files, collection_name).await.unwrap();
        });
    });
}

#[cfg(feature = "heavyweight")]
pub async fn embed_and_index(
    files: &str,
    collection_name: &str,
    model: &str,
    variant: &str,
    batch_size: u64,
    index_columns: &[String],
    hf_token: Option<String>,
) -> anyhow::Result<()> {
    let mut config = CollectionConfig::default();
    config.name = collection_name.to_string();
    config.index_columns = index_columns.to_vec();
    config.model_name = model.to_string();
    config.model_variant = variant.to_string();

    let model_manager = ModelManagerActor::new().start();
    let collection_manager = CollectionManagerActor::new(hf_token, model_manager.clone()).start();

    let collection_addr = collection_manager
        .send(CreateCollection {
            config,
            overwrite: true,
        })
        .await??;

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
    }

    // Embed columns
    // We need to fetch the model_id
    let model_id = collection_manager
        .send(GetModelIdForCollection {
            name: collection_name.to_string(),
        })
        .await??;

    for column_name in index_columns {
        collection_addr
            .send(EmbedColumn {
                name: column_name.clone(),
                batch_size,
                model_id,
            })
            .await??;
    }
    Ok(())
}

#[cfg(feature = "heavyweight")]
fn benchmark_pipeline(c: &mut Criterion) {
    let runner = tokio::runtime::Runtime::new().unwrap();
    c.bench_function("embed_and_index", |b| {
        b.to_async(&runner).iter(|| async {
            let files = "./benches/rag_instruct_benchmark_tester.jsonl";
            let collection_name = "test_collection";
            let model = "hf://mys/minilm";
            let variant = "f32";
            let batch_size = 32;
            let index_columns = vec![
                "context".to_string(),
                "query".to_string(),
                "answer".to_string(),
            ];
            let hf_token = None;

            embed_and_index(
                files,
                collection_name,
                model,
                variant,
                batch_size,
                &index_columns,
                hf_token,
            )
            .await
            .unwrap();
        });
    });
}

#[cfg(not(feature = "heavyweight"))]
criterion_group!(benches, benchmark_import_jsonl);
#[cfg(feature = "heavyweight")]
criterion_group!(benches, benchmark_pipeline, benchmark_import_jsonl);
criterion_main!(benches);
