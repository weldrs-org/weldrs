//! Error types for the weldrs crate.

/// The main error type for weldrs operations.
#[derive(thiserror::Error, Debug)]
pub enum WeldrsError {
    /// An error originating from the Polars engine.
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    /// Invalid or incomplete configuration (e.g. no comparisons defined).
    #[error("Configuration error: {0}")]
    Config(String),
    /// An error during model training (EM, u-estimation, lambda estimation).
    #[error("Training error: {0}")]
    Training(String),
    /// JSON serialization or deserialization failed.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// A convenience alias for `std::result::Result<T, WeldrsError>`.
pub type Result<T> = std::result::Result<T, WeldrsError>;
