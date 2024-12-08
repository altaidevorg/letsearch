extern crate letsearch;
use anyhow;
use criterion::{criterion_group, criterion_main, Criterion};
use letsearch::collection::collection_type::Collection;
use letsearch::collection::collection_utils::CollectionConfig;

#[cfg(feature = "heavyweight")]
use letsearch::collection::collection_manager::CollectionManager;

use tokio::runtime::Runtime;

pub async fn import_jsonl(files: &str, collection_name: &str) -> anyhow::Result<()> {
    let mut config = CollectionConfig::default();
    config.name = collection_name.to_string();

    let collection = Collection::new(config, true).await?;
    collection.import_jsonl(files).await?;
    Ok(())
}

fn benchmark_import_jsonl(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    c.bench_function("import_jsonl", |b| {
        b.to_async(&rt).iter(|| async {
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

    let collection_manager = CollectionManager::new(hf_token);
    collection_manager.create_collection(config, true).await?;
    if files.ends_with(".jsonl") {
        collection_manager
            .import_jsonl(collection_name, files)
            .await?;
    } else if files.ends_with(".parquet") {
        collection_manager
            .import_parquet(collection_name, files)
            .await?;
    }
    for column_name in index_columns {
        collection_manager
            .embed_column(collection_name, column_name, batch_size)
            .await?;
    }
    Ok(())
}

#[cfg(feature = "heavyweight")]
fn benchmark_pipeline(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    c.bench_function("embed_and_index", |b| {
        b.to_async(&rt).iter(|| async {
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
