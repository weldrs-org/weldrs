//! Error types for the weldrs crate.
//!
//! All fallible operations in weldrs return [`Result<T>`], an alias for
//! `std::result::Result<T, WeldrsError>`.
//!
//! # Error variants
//!
//! | Variant | Typical source |
//! |---------|---------------|
//! | [`WeldrsError::Polars`] | Underlying Polars engine failures (join, collect, schema) |
//! | [`WeldrsError::Config`] | Invalid configuration (no comparisons, missing columns) |
//! | [`WeldrsError::Training`] | Training-time problems (too few records, EM failures) |
//! | [`WeldrsError::Serde`] | JSON serialization / deserialization of settings |
//! | [`WeldrsError::Visualization`] | Chart rendering failures (requires the `visualize` feature) |

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
    #[error("{stage}: {message}")]
    Training {
        /// Which pipeline stage produced the error (e.g. "em", "predict",
        /// "blocking", "clustering", "estimate_u", "estimate_lambda").
        stage: &'static str,
        /// Human-readable description of what went wrong.
        message: String,
    },
    /// JSON serialization or deserialization failed.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    /// An error during chart rendering.
    #[error("Visualization error: {0}")]
    Visualization(String),
}

/// A convenience alias for `std::result::Result<T, WeldrsError>`.
pub type Result<T> = std::result::Result<T, WeldrsError>;
