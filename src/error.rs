use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProjectError {
    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),
    #[error("Model with ID '{0}' not found")]
    ModelNotFound(u32),
    #[error("Database error: {0}")]
    DatabaseError(#[from] duckdb::Error),
    // ... other error variants
}
