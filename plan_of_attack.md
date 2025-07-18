# Plan of Attack: Refactoring `letsearch` to an Actor Model (Revised)

This document provides a detailed, step-by-step plan to refactor the `letsearch` project from its current shared-state concurrency model (`Arc<RwLock<T>>`) to a more scalable and robust actor-based model using the `actix` framework.

## 1. Guiding Principles & Goals

-   **Scalability:** Eliminate lock contention by replacing shared-state `RwLock`s with message passing between isolated actors. This will improve throughput under heavy concurrent loads.
-   **State Management:** Each actor will own and manage its own state, making the flow of data explicit and easier to reason about.
-   **Robustness:** Reduce the possibility of deadlocks and introduce a structured error handling system via message responses.
-   **Performance:** Carefully handle CPU-bound and blocking I/O tasks by offloading them to a dedicated thread pool to avoid stalling the async actor system.

We will use the **`actix`** crate, as the project already uses `actix-web`, ensuring ecosystem compatibility.

## 2. Pre-flight Checks & Key Architectural Decisions

Before implementation, we must address two critical design points discovered during this review.

### 2.1. Handling of Blocking/CPU-Bound Operations

The current code correctly uses `tokio::task::spawn_blocking` for ONNX inference and `rayon` for vector indexing. This pattern is essential and must be preserved within the actor model. **An actor's handler should never perform a long-running, blocking operation directly on its own execution context.**

-   **Decision:** All CPU-bound tasks (`vector_index.add`, `vector_index.search`) and potentially blocking I/O (like DuckDB queries) will be wrapped in `tokio::task::spawn_blocking` inside the actor's message handlers.

### 2.2. Handling of `!Send` Types (DuckDB Connection)

A `duckdb::Connection` is `!Send`, meaning it cannot be safely sent between threads. This poses a problem for the standard Actix actor model, where an actor and its state might be moved between threads by the runtime.

-   **Decision:** The `CollectionActor` will **not** hold a `duckdb::Connection` in its state. Instead, it will hold the `db_path: PathBuf`. Each message handler that needs to interact with the database will open a *new connection* inside a `spawn_blocking` call. This ensures the `!Send` connection never leaves the thread it was created on, making the actor `Send`.

    ```rust
    // Example pattern for DuckDB operations inside a CollectionActor handler
    let db_path = self.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let conn = Connection::open(&db_path)?;
        // ... perform all database operations here ...
        conn.execute(...)
    }).await??; // .await? (for JoinError) and ?? (for the inner Result)
    ```

## 3. High-Level Actor Design

-   **`ModelManagerActor`:** A singleton actor responsible for the lifecycle of ML models. Replaces `ModelManager`.
-   **`CollectionActor`:** An actor for each collection. It will own its configuration and vector indices. Replaces the `Collection` struct.
-   **`CollectionManagerActor`:** A singleton actor that acts as a registry for `CollectionActor`s. Replaces `CollectionManager`.

## 4. Detailed Refactoring Plan

### Phase 1: Setup and Dependencies

1.  **Add `actix` and `futures` to `Cargo.toml`:**
    ```toml
    [dependencies]
    actix = "0.13"
    futures = "0.3" # Useful for the block_on executor if needed.
    # ... other dependencies
    ```
2.  **Create a new module for actors:** Create `src/actors/mod.rs` and corresponding files (`model_actor.rs`, `collection_actor.rs`, `manager_actor.rs`).
3.  **Define a unified Error Type:** Create a project-wide error enum to be used in message results for cleaner error handling than `anyhow::Error`.
    ```rust
    // In a new file, e.g., src/error.rs
    #[derive(thiserror::Error, Debug)]
    pub enum ProjectError {
        #[error("Collection '{0}' not found")]
        CollectionNotFound(String),
        #[error("Model with ID '{0}' not found")]
        ModelNotFound(u32),
        #[error("Database error: {0}")]
        DatabaseError(#[from] duckdb::Error),
        // ... other error variants
    }
    ```

### Phase 2: Actor and Message Definitions

#### 2.1. `ModelManagerActor` (`src/actors/model_actor.rs`)
-   **State:** `models: HashMap<u32, Arc<dyn ONNXModel>>`, `next_id: u32`.
-   **Messages:**
    -   `LoadModel { path, variant, token }` -> `Result<u32, ProjectError>`
    -   `Predict { id, texts }` -> `Result<Embeddings, ProjectError>`
    -   `GetModelMetadata { id }` -> `Result<(i64, ModelOutputDType), ProjectError>`

#### 2.2. `CollectionActor` (`src/actors/collection_actor.rs`)
-   **State:** `config: CollectionConfig`, `db_path: PathBuf`, `vector_indices: HashMap<String, VectorIndex>`, `model_manager: Addr<ModelManagerActor>`.
-   **Messages:**
    -   `ImportJsonl { path }` -> `Result<(), ProjectError>`
    -   `ImportParquet { path }` -> `Result<(), ProjectError>`
    -   `EmbedColumn { name, batch_size }` -> `Result<(), ProjectError>`
    -   `Search { column, query, limit }` -> `Result<Vec<SearchResult>, ProjectError>`
    -   `GetConfig` -> `Result<CollectionConfig, ProjectError>`

#### 2.3. `CollectionManagerActor` (`src/actors/manager_actor.rs`)
-   **State:** `collections: HashMap<String, Addr<CollectionActor>>`, `model_manager: Addr<ModelManagerActor>`, `hf_token: Option<String>`.
-   **Messages:**
    -   `CreateCollection { config, overwrite }` -> `Result<Addr<CollectionActor>, ProjectError>`
    -   `LoadCollection { name }` -> `Result<Addr<CollectionActor>, ProjectError>`
    -   `GetCollectionAddr { name }` -> `Result<Addr<CollectionActor>, ProjectError>`
    -   `GetAllCollectionConfigs` -> `Result<Vec<CollectionConfig>, ProjectError>`

### Phase 3: Step-by-Step Implementation

#### Step 3.1: Refactor Model Trait and Implement `ModelManagerActor`
1.  **Simplify `ONNXModelTrait`:** Make its methods **synchronous**. The actor will manage the async execution context.
    ```rust
    // src/model/model_utils.rs
    // No more `#[async_trait]` here
    pub trait ONNXModelTrait {
        fn predict_f32(&self, texts: Vec<&str>) -> anyhow::Result<Arc<Array2<f32>>>;
        // ... other methods are sync ...
    }
    ```
2.  **Update `BertONNX`:** Remove `async` and `spawn_blocking` from the `predict_f32`/`f16` implementations. They will now be simple, blocking functions.
3.  **Implement `ModelManagerActor`:**
    -   In the `Predict` handler, wrap the call to the synchronous `model.predict_f32` in `spawn_blocking`. This correctly isolates the blocking code.

#### Step 3.2: Implement `CollectionActor`
1.  Implement the actor and its message handlers in `src/actors/collection_actor.rs`.
2.  For `ImportJsonl`/`Parquet` handlers, use the `spawn_blocking` pattern with `Connection::open()` as described in **Section 2.2**.
3.  Implement the `EmbedColumn` handler:
    -   First, `await` a `GetModelMetadata` message to the `model_manager` to get the vector dimension and data type.
    -   Use `spawn_blocking` to query DuckDB for the total row count for progress calculation.
    -   Loop through batches. In each loop iteration:
        -   Use `spawn_blocking` to fetch a batch of data and keys from DuckDB.
        -   `await` a `Predict` message to the `model_manager`.
        -   Use `spawn_blocking` again to call the CPU-intensive `vector_index.add()`.
4.  Implement the `Search` handler, following the same pattern: `await` prediction from the model actor, then `spawn_blocking` for the vector search, and finally `spawn_blocking` for retrieving data from DuckDB.

#### Step 3.3: Implement `CollectionManagerActor`
1.  This actor is mostly a stateful registry. Its handlers will be straightforward.
2.  The `CreateCollection` handler will instantiate a `CollectionActor`, providing it with the address of the `ModelManagerActor`, and then call `.start()` on it.
3.  The `LoadCollection` handler will do the same but will first need to load the config from disk to initialize the `CollectionActor`.

#### Step 3.4: Update Web Server (`src/serve.rs`)
1.  In `run_server`, start the actor system.
    ```rust
    // In run_server
    let model_manager_addr = ModelManagerActor::new().start();
    let collection_manager_addr = CollectionManagerActor::new(token, model_manager_addr.clone()).start();
    
    // Await the loading of the initial collection to ensure server is ready
    let load_result = collection_manager_addr.send(LoadCollection { name: collection_name }).await;
    if load_result.is_err() || load_result.unwrap().is_err() {
        // Handle error and exit if collection can't be loaded
        panic!("Failed to load initial collection.");
    }

    let shared_manager_addr = web::Data::new(collection_manager_addr);
    // ... start HttpServer ...
    ```
2.  Replace `app_data` with `web::Data<Addr<CollectionManagerActor>>`.
3.  Rewrite all HTTP handlers to be async and use `addr.send(...).await`. The `search` handler example from the previous plan is a good template.

#### Step 3.5: Update CLI (`src/main.rs`)
1.  The `Index` command logic will be wrapped in `actix::System::new().block_on(async { ... });`.
2.  Inside this block, it will start the actors and send a sequence of messages, `await`ing each one to ensure sequential execution (create -> import -> embed).

### Phase 4: Testing and Validation
1.  **Update Unit Tests:** Rewrite tests for `Collection` and `ModelManager` to test the new actors. This will involve creating a test `System` for each test function.
2.  **Update Integration Tests & Benchmarks:** Adapt the `cargo bench` and `cargo test --features heavyweight` tests to the new actor-based architecture.
3.  **Benchmarking:** Run benchmarks before and after the refactor to measure the impact on throughput and latency, especially under concurrent load (using a tool like `wrk` or `oha` against the search endpoint).
4.  **New Tests:** Add tests for actor supervision and lifecycle (e.g., ensuring actors restart or fail gracefully).

This revised plan is more detailed and addresses critical technical challenges discovered during the review, particularly regarding the handling of `!Send` types and the management of blocking operations. It provides a clearer and safer path to a successful refactoring.