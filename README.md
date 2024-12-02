## roadmap
1. Refactor model manager to use string keys in hash map.
2. Maybe export a `ModelInfo` struct with `get_models()`?
3. Chunking support and multiple vectors per key.
4. maybe get /schema endpoint
5. incremental index building, maybe postponed later.
6. llama.cpp backend
7. automatically get letsencrypt TLS certificate and API key
8. finetune embedding models

## TODO for search
1. Allow multiple column index

## Test command
```sh
cargo run -- index -m ../altaidemo/model/minilm -c test1 hf://datasets/neural-bridge/rag-dataset-12000/**/*.parquet --overwrite -b 128 -i context
```

```sh
cargo run -- index -c test2 -m ../altaidemo/model/minilm -b 128 hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/*.parquet -i passage --overwrite
```
