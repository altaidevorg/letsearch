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

## Task 2: Refactor Model Trait and Implement ModelManagerActor

The next step is to refactor the model traits to be synchronous and to implement the `ModelManagerActor`. This involves:

1.  **Refactor `ONNXModelTrait`**: Make the methods in `ONNXModelTrait` synchronous by removing the `async_trait` macro and the `async` keyword.
2.  **Update `BertONNX`**: Update the `BertONNX` implementation to match the new synchronous trait definitions. This includes removing `async` and `spawn_blocking` from the `predict_f32`/`f16` implementations.
3.  **Implement `ModelManagerActor`**: Create the `ModelManagerActor` in `src/actors/model_actor.rs`. This actor will be responsible for managing the lifecycle of the models and will use `spawn_blocking` to run the synchronous `predict` methods in a non-blocking way.
