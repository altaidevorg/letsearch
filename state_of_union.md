# State of the Union: `letsearch` Architecture

This document outlines the current architecture of the `letsearch` project before the planned refactoring to an actor-based model. The system is a monolithic binary that provides both CLI for data management and a web server for search queries.

## Core Concurrency Model: Shared State with `Arc<RwLock<T>>`

The entire concurrency strategy is built upon Rust's standard shared-state model.

-   **`Arc<T>` (Atomically Reference-Counted):** This is used extensively to allow multiple parts of the application to have shared ownership of key components without copying the underlying data. For example, a single `CollectionManager` is shared across all web server threads, and multiple threads can reference the same `Collection` or `ModelManager`.
-   **`tokio::sync::RwLock<T>` (Asynchronous Read-Write Lock):** This is the primary mechanism for ensuring thread safety. It allows for either multiple concurrent readers or a single exclusive writer. This pattern is used to wrap almost every shared component, including the central `CollectionManager`, individual `Collection`s, and even the maps holding these components.

This model is straightforward to implement but has scalability limitations. High contention on a lock (many threads trying to write at once) can lead to performance degradation as threads must wait. Long-running operations inside a write lock can starve all other readers and writers for that resource.

## Key Architectural Components

The architecture can be broken down into the following layers and components:

### 1. Entrypoints

The application has two primary entry points, both defined in `src/main.rs`.

-   **CLI (`clap`):** The `Index`, and `ListModels` commands are handled via a `clap`-based CLI. These are typically one-off, synchronous-style operations that instantiate a `CollectionManager`, perform a series of tasks (create, import, embed), and then exit.
-   **Web Server (`actix-web`):** The `Serve` command starts a persistent Actix Web server defined in `src/serve.rs`. This server is designed for concurrent request handling.

### 2. Orchestration Layer

-   **`CollectionManager` (`src/collection/collection_manager.rs`):** This struct acts as a **Facade** for the entire system. It is the main point of interaction for both the CLI and the web server.
    -   **State:** It holds a `RwLock<HashMap<String, Arc<RwLock<Collection>>>>`, which is the central registry of all active collections. It also holds an `Arc<RwLock<ModelManager>>` to manage embedding models.
    -   **Function:** It routes requests to the appropriate `Collection` instance. For example, when `embed_column` is called, it looks up the collection `Arc`, acquires a write lock on that specific `Collection`, and then calls the method on it. It also coordinates with the `ModelManager` to ensure models are loaded before they are needed.

### 3. Core Data & Logic Unit

-   **`Collection` (`src/collection/collection_type.rs`):** This struct represents a single, self-contained dataset.
    -   **State:**
        -   `config: CollectionConfig`: Holds all metadata.
        -   `conn: Arc<RwLock<Connection>>`: A thread-safe handle to a **DuckDB** database connection, used for storing and retrieving the original document data.
        -   `vector_index: RwLock<HashMap<String, Arc<RwLock<VectorIndex>>>>`: A map from column names to their corresponding vector search indices.
    -   **Function:** This is where the heavy lifting occurs.
        -   `import_jsonl`/`import_parquet`: Loads data into DuckDB.
        -   `embed_column`: A long-running, CPU-bound operation. It fetches data in batches from DuckDB, sends it to the `ModelManager` for embedding, and then adds the resulting vectors to the appropriate `VectorIndex`.
        -   `search`: Also CPU-bound. It takes a query, gets its embedding from the `ModelManager`, searches the `VectorIndex`, and then retrieves the original data for the top results from DuckDB.

### 4. Model Management

-   **`ModelManager` (`src/model/model_manager.rs`):** Implements a **Flyweight** pattern for ML models.
    -   **State:** Holds a `RwLock<HashMap<u32, Arc<RwLock<dyn ONNXModel>>>>`. This ensures that each unique model is loaded into memory only once and is identified by a unique ID.
    -   **Function:** It abstracts the model loading (from local disk or Hugging Face via `hf_ops`) and prediction process.
-   **`BertONNX` (`src/model/backends/onnx/bert_onnx.rs`):** The concrete implementation of an `ONNXModel`.
    -   **Function:** It encapsulates all the logic for interacting with the `ort` (ONNX Runtime) and `tokenizers` crates.
    -   **CPU-Bound Work Handling:** Crucially, the `predict_f32` and `predict_f16` methods wrap the CPU-intensive tokenization and model inference calls in `tokio::task::spawn_blocking`. This is a key optimization that prevents these long-running tasks from blocking the main Tokio runtime threads, allowing the server to remain responsive.

### 5. Vector Indexing

-   **`VectorIndex` (`src/collection/vector_index.rs`):** An abstraction over the `usearch` library.
    -   **Function:** Manages the lifecycle (create, load, save) of a single vector index.
    -   **CPU-Bound Work Handling:** The `add` method uses the `rayon` crate (`par_iter`) to parallelize the process of adding vectors to the index across multiple CPU cores. This is another important optimization for speeding up the indexing process.

## Summary of Concurrency and Performance

-   **Web Server (`serve`):** The Actix Web server shares the `CollectionManager` via `web::Data::new(RwLock::new(collection_manager))`. Every incoming HTTP request that needs to perform a search must acquire a *read* lock on the `CollectionManager`, then a *read* lock on the specific `Collection`, and so on. While reads are concurrent, any write operation (like a hypothetical `/add-documents` endpoint) would require exclusive write locks, potentially blocking all other operations.
-   **Indexing (`index`):** This is a single-threaded process from a user's perspective, but it leverages parallelism internally. `embed_column` processes batches sequentially, but within each batch, `BertONNX` uses `spawn_blocking` for inference, and `VectorIndex` uses `rayon` for adding to the index.
-   **Bottlenecks:** The primary architectural bottleneck is the nested `RwLock` structure. Under heavy concurrent write load (if such features were added) or even with many concurrent long-running search queries, lock contention could become a significant issue, limiting throughput. A search query holds read locks for a considerable duration, which can delay write locks from being acquired.

This architecture is a solid, idiomatic Rust implementation for a project of this scale. However, the user's goal to refactor to an actor model is a logical next step for enhancing scalability and managing component state more explicitly.
