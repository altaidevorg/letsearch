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

## 🏎️ Quickstart
1. Download the latest prebuilt binary from [releases](https://github.com/monatis/letsearch/releases).
2. And simply run it on terminal:

```sh
./letsearch
```

Wuhu! Now you already know how to use letsearch! 🙋 It's that simple.

⚠️ **Note**: letsearch is at a early stage of development, so rapid changes in the API should be accepted.

## 🧮 Models
- To see the models currently available on HuggingFace Hub, run:

```sh
./letsearch list-models
```

To convert your own models to a format that you can use with letsearch, see [this script](./scripts/export_to_onnx.py).

## 🔍 Search
Se [this](./scripts/test.py) for a dead simple request example. A full Python client is on the way.

## 🧭 roadmap
letsearch is an early-stage solution, but it already has a concrete roadmap to make RAG uncool again.

You can check the following items in the current agenda and give a 👍 to the issue of the feature that you particularly find useful for your use case.
The most popular features will prioritized.

If you have something in mind that you think will be a great addition to letsearch, please let me know by raising an [issue](https://github.com/monatis/letsearch/issues/new).

- [ ] Incremental index building: appending on terminal and `/add` endpoint on API (#9)
- [ ] Import content from PDFs and automatic chunking support (#10)
- [ ] MCP support (#11)
- [ ] llama.cpp backend (#12)
- [ ] Multimodal support (#13)
- [ ] Support API key (#14)

Please also check [other issues](https://github.com/monatis/letsearch/issues).

## License
letsearch is distributed under the terms of [the Apache License 2.0](https://github.com/monatis/letsearch).
