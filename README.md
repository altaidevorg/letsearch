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
