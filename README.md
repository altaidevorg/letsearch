# letsearch

![logo](./assets/logo.jpg)

A vector DB so easy, even your grandparents can build a RAG system 😁

## ❓ What is this?

letsearch is a single executable binary to easily embed, index and search your documents without writing a single line of code. It's a RAG-native vector database to make your documents available for AI as quickly as possible.

With its built-in support for ONNX inference (llama.cpp and GGUF support coming soon!), you can import, embed and index your documents from JSONL and Parquet files --it can even fetch them from HuggingFace Hub for you (PDF / Doc / Dox support coming soon with automatic chunking feature!).

[MCP](https://modelcontextprotocol.io/introduction) support is also on the way!

## 🖼️ Features

- Import documents from JSONL files.
- Import documents from Parquet files.
- Import datasets from Huggingface Hub only with `hf://datasets/*` path.
- Automatically create a collection and and index multiple columns at once with the given embedding model.
- Download models from HuggingFace Hub automatically only with a path `hf://*`.
- List models available on HuggingFace Hub.
- Convert and bring your own models.
- Upload and/or download prebuilt collections on HuggingFace Hub easily (coming soon).

## 😕 Why does it exists?

Building RAG (Retrieval-Augmented Generation) or semantic search applications often involves dealing with the complexities of vector operations embedding management, and infrastructure setup. `letsearch` was created to eliminate these burdens and streamline the process of building and serving vector indexes.

### Key Benefits

- **No More Vector Ops Hassle**  
  Focus on your application logic without worrying about the intricacies of vector indexing, storage, or retrieval.

- **Simplified Collection Management**  
  Easily create, manage, and share collections of embeddings, whether from JSONL, Parquet, or even HuggingFace datasets.

- **From Experimentation to Production in No Time**  
  Drastically reduce the time required to go from prototyping your RAG or search workflows to serving requests in production.

- **Say Goodbye to Boilerplate**  
  Avoid repetitive setup and integration code. `letsearch` provides a single, ready-to-run binary to embed, index, and search your documents. This is particularly useful for serverless cloud jobs and local AI applications.

By combining these advantages with built-in support for ONNX models and plans for multimodal / multibackend capabilities, `letsearch` is your go-to tool for making documents AI-ready in record time.

## 🏎️ Quickstart

1. Download the latest prebuilt binary from [releases](https://github.com/monatis/letsearch/releases).
2. And simply run it on terminal:

```sh
./letsearch
```

Wuhu! Now you already know how to use letsearch! 🙋 It's that simple.

⚠️ **Note**: letsearch is at a early stage of development, so rapid changes in the API should be expected.

## 🚧 Indexing documents

```sh
./letsearch index --collection-name test1 --index-columns context hf://datasets/neural-bridge/rag-dataset-1200/**/*.parquet
```

With a single CLI command, you:

- downloaded `.parquet` files from [a HF dataset repository](https://huggingface.co/datasets/neural-bridge/rag-dataset-1200/).
- downloaded [a model from HF Hub](https://huggingface.co/mys/minilm).
- imported your documents to the DB.
- embedded texts in the column `context`.
- built a vector index.

You can use local or `hf://` paths to import your documents in `.jsonl` or `.parquet` files.
Regular paths and/or glob patterns are supported.

Run:

```sh
./letsearch index --help
```

for more usage tips.

## 🔍 Search

Use the same binary to serve your index:

```sh
./letsearch serve -c test1
```

Then, it's quite easy to make search requests with [letsearch-client](https://github.com/monatis/letsearch-client).

## 🧮 Models

- To see the models currently available on HuggingFace Hub, run:

```sh
./letsearch list-models
```

To convert your own models to a format that you can use with letsearch, see [letsearch-client](https://github.com/monatis/letsearch-client).

## 🧭 roadmap

letsearch is an early-stage solution, but it already has a concrete roadmap to make RAG uncool again.

You can check the following items in the current agenda and give a 👍 to the issue of the feature that you particularly find useful for your use case.
The most popular features will prioritized.

If you have something in mind that you think will be a great addition to letsearch, please let me know by raising an [issue](https://github.com/monatis/letsearch/issues/new).

- [ ] [Incremental index building: appending on terminal and `/add` endpoint on API](https://github.com/monatis/letsearch/issues/9)
- [ ] [Import content from PDFs and automatic chunking support](https://github.com/monatis/letsearch/issues/10)
- [ ] [MCP support](https://github.com/monatis/letsearch/issues/11)
- [ ] [llama.cpp backend](https://github.com/monatis/letsearch/issues/12)
- [ ] [Multimodal support](https://github.com/monatis/letsearch/issues/13)
- [ ] [Support API key](https://github.com/monatis/letsearch/issues/14)

Please also check [other issues](https://github.com/monatis/letsearch/issues).

## 🌡️ Tests and Benchmarks

```sh
cargo bench
```

To benchmark the full pipeline, you can also run:

**Note**: This can take a lot of time.

```sh
cargo bench --feature heavyweight
```

To run the tests:

```sh
cargo test
```

## 📖 License

letsearch is distributed under the terms of [the Apache License 2.0](https://github.com/monatis/letsearch).
