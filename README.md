# letsearch
![logo](./assets/logo.jpg)

A single binary to embed, index, search and serve your documents

## What is this?
letsearch is a single executable binary to easily embed, index and search your documents without writing a single line of code. With its built-in support for ONNX inference (llama.cpp and GGUF support coming soon!), you can import, embed and index your documents from JSONL and Parquet files --it can even fetch them from HuggingFace Hub for you (PDF / Doc / Dox support coming soon with automatic chunking feature!).

[MCP](https://modelcontextprotocol.io/introduction) support is also on the way!

## Usage
1. Download the latest prebuilt binary from [releases](https://github.com/monatis/letsearch/releases).
2. And simply run on terminal:

```sh
./letsearch
```

Wuhu! Now you already know how to use letsearch! It's that simple.


## Models
- To see the models currently available on HuggingFace Hub, run:

```sh
./letsearch list-models
```
To convert your own models to a format that you can use with letsearch, see [this script](./scripts/export_to_onnx.py).

## Search
Se [this](./scripts/test.py) for a dead simple request example. A full Python client is on the way.

## roadmap
- [ ] Incremental index building (appending on terminal and `/add` endpoint on API)
- [ ] Chunking support and multiple vectors per key.
- [ ] MCP support.
- [ ] llama.cpp backend
- [ ] automatically get letsencrypt TLS certificate and API key
- [ ] finetune embedding models
