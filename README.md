# letsearch
![logo](./assets/logo.jpg)

Let's make RAG uncool again with a single binary to embed, index, search and serve your documents

## What is this?
letsearch is a single executable binary to easily embed, index and search your documents without writing a single line of code. It's a RAG-native vector database to make your documents available for AI as quickly as possible.

With its built-in support for ONNX inference (llama.cpp and GGUF support coming soon!), you can import, embed and index your documents from JSONL and Parquet files --it can even fetch them from HuggingFace Hub for you (PDF / Doc / Dox support coming soon with automatic chunking feature!).

[MCP](https://modelcontextprotocol.io/introduction) support is also on the way!

## Features
- Import documents from JSONL files.
- Import documents from Parquet files.
- Import datasets from Huggingface Hub only with `hf://datasets/*` path.
- Automatically create a collection and and index multiple columns at once with the given embedding model.
- Download models from HuggingFace Hub automatically only with a path `hf://*`.
- List models available on HuggingFace Hub.
- Convert and bring your own models.
- Upload and/or download prebuilt collections on HuggingFace Hub easily (coming soon).

## Quickstart
1. Download the latest prebuilt binary from [releases](https://github.com/monatis/letsearch/releases).
2. And simply run on terminal:

```sh
./letsearch
```

Wuhu! Now you already know how to use letsearch! It's that simple.

**Note**: letsearch is at a early stage of development, so rapid changes in the API should be accepted.

## Models
- To see the models currently available on HuggingFace Hub, run:

```sh
./letsearch list-models
```

To convert your own models to a format that you can use with letsearch, see [this script](./scripts/export_to_onnx.py).

## Search
Se [this](./scripts/test.py) for a dead simple request example. A full Python client is on the way.

## roadmap
letsearch is an early-stage solution, but it already has a concrete roadmap to make RAG uncool again.

- [ ] Incremental index building (appending on terminal and `/add` endpoint on API)
- [ ] Automatic chunking support and multiple vectors per key.
- [ ] MCP support.
- [ ] llama.cpp backend
- [ ] Multimodal support
- [ ] automatically get letsencrypt TLS certificate and API key
- [ ] finetune embedding models
