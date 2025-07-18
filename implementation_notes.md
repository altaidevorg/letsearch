# Implementation Notes: Refactoring `letsearch` to an Actor Model

This document outlines the step-by-step process of refactoring `letsearch` from a shared-state concurrency model to an actor-based model using `actix`.

## Guiding Principles

- **Incremental Refactoring**: Each step should result in a compilable and functional state of the project. This allows for continuous testing and validation.
- **Coexistence of Models**: During the transition, both the shared-state and actor models will coexist. We will gradually migrate components to the actor model.
- **Start Small**: We will start with the simplest and most independent components to minimize the risk of breaking the application.

## Task 1: Initial Setup (Completed)

The first task was to set up the basic infrastructure for the actor model without modifying any existing logic. This included:

1.  **Add Dependencies**: Added `actix`, `futures`, and `thiserror` to `Cargo.toml`.
2.  **Create Actor Module**: Created a new module `src/actors` to house the actor-related code.
3.  **Create Error Module**: Created a new module `src/error` for a unified error type.

This task is complete and the project is in a compilable state.

## Task 2: Refactor Model Trait and Implement ModelManagerActor (Completed)

This task focused on creating the `ModelManagerActor` to handle the lifecycle and prediction requests for embedding models.

1.  **Refactor `ONNXModelTrait`**: The methods in `ONNXModelTrait` were made synchronous, removing `async_trait`.
2.  **Update `BertONNX`**: The `BertONNX` implementation was updated to match the synchronous trait, with blocking `predict` methods.
3.  **Implement `ModelManagerActor`**: The `ModelManagerActor` was created in `src/actors/model_actor.rs`. Its handlers for `Predict` messages correctly wrap the synchronous, CPU-bound model inference calls in `tokio::task::spawn_blocking` to avoid blocking the actor's thread.

## Task 3: Implement CollectionActor (Completed)

This task involved creating the `CollectionActor` to manage the state and operations for a single collection.

1.  **Actor and Message Definition**: The `CollectionActor` and its associated messages were defined in `src/actors/collection_actor.rs`.
2.  **State Management**: The actor's state and handlers were designed to be compatible with the `actix` framework, particularly by handling `!Send` types correctly and managing shared access to vector indices.
3.  **Handler Implementation**: All message handlers (`ImportJsonl`, `ImportParquet`, `EmbedColumn`, `Search`, `GetConfig`) were implemented.

## Task 4: Implement CollectionManagerActor (Completed)

This task involved creating the `CollectionManagerActor` to serve as the top-level orchestrator and registry for all `CollectionActor` instances.

1.  **Actor and Message Definition**: The `CollectionManagerActor` and its messages (`CreateCollection`, `LoadCollection`, `GetCollectionAddr`, `GetAllCollectionConfigs`) were defined in `src/actors/collection_manager_actor.rs`.
2.  **Handler Implementation**: All message handlers were implemented, providing the logic for creating, loading, and retrieving collection actors.

## Task 5: Integrate Actors into Web Server (Completed)

This task involved replacing the old shared-state `CollectionManager` in the web server (`src/serve.rs`) with the new actor system.

1.  **Update `run_server` function**: The `run_server` function was updated to instantiate and start the `ModelManagerActor` and `CollectionManagerActor`, and to load the initial collection.
2.  **Rewrite HTTP Handlers**: The HTTP handlers were rewritten to send messages to the actor system and to handle the responses.

## Task 6: Integrate Actors into CLI (Completed)

The final task was to update the `Index` command in the CLI (`src/main.rs`) to use the actor system.

1.  **Update `main` function**: The `main` function was updated to use `actix::main`.
2.  **Update `Index` command handler**: The `Index` command handler was rewritten to start the actor system and to send messages for each step of the indexing process (creating a collection, importing data, and embedding columns).

The refactoring to an actor-based model is now complete. The project is in a compilable and functional state, and the core components have been migrated to the new architecture.