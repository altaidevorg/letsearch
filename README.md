## roadmap
1. specify which columns to index on the command line.
2. index json and parquet files. they should be imported to duckdb tables.
3. Refactor model manager to use string keys in hash map.
4. Maybe export a `ModelInfo` struct with `get_models()`?
5. `serve` subcommand should expose a simple API for search.
6. Chunking support and multiple vectors per key.
7. maybe get /schema endpoint
8. incremental index building, maybe postponed later.
9. llama.cpp backend
10. automatically get letsencrypt TLS certificate and API key
11. finetune embedding models

## TODO for search
1. Add `model_name` field to `CollectionConfig`.
2. Add `requested_models() -> Vec<String>` to `Collection` to return the model names to load.
3. Introduce a `ModelManager` field to `CollectionManager` and update `CollectionManager::load_collection()` to load models requested by `Collection`.
4. implement `CollectionManager::search(collection_name: String, column_name: String, num_results)` that will call Collection::search(model_manager, model_id, column_name, query, num_results)