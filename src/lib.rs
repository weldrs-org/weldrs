//! # weldrs
//!
//! Fellegi-Sunter probabilistic record linkage in Rust, powered by
//! [Polars](https://pola.rs/).
//!
//! **weldrs** provides an end-to-end pipeline for deduplicating or linking
//! records across datasets:
//!
//! 1. **Define comparisons** — specify how columns are compared (exact match,
//!    Jaro-Winkler, Levenshtein, etc.) using [`comparison::ComparisonBuilder`].
//! 2. **Configure settings** — choose a link type, add blocking rules, and
//!    build a [`settings::Settings`] value.
//! 3. **Train** — estimate model parameters (lambda, u-probabilities, m/u via
//!    EM) through the [`linker::Linker`] orchestrator.
//! 4. **Predict** — score candidate pairs with Fellegi-Sunter match weights
//!    and probabilities.
//! 5. **Cluster** — group linked records into clusters using connected
//!    components.
//!
//! Most users will interact with the library through the [`prelude`] module
//! and the [`linker::Linker`] struct.
//!
//! # Quick example
//!
//! ```no_run
//! use polars::prelude::*;
//! use weldrs::comparison::ComparisonBuilder;
//! use weldrs::prelude::*;
//!
//! fn main() -> Result<()> {
//!     let df = df!(
//!         "unique_id"  => [1i64, 2, 3, 4],
//!         "first_name" => ["John", "Jane", "Jon", "Jane"],
//!         "surname"    => ["Smith", "Doe", "Smith", "Doe"],
//!     )?;
//!
//!     let settings = Settings::builder(LinkType::DedupeOnly)
//!         .comparison(
//!             ComparisonBuilder::new("first_name")
//!                 .null_level()
//!                 .exact_match_level()
//!                 .jaro_winkler_level(0.88)
//!                 .else_level()
//!                 .build(),
//!         )
//!         .comparison(
//!             ComparisonBuilder::new("surname")
//!                 .null_level()
//!                 .exact_match_level()
//!                 .else_level()
//!                 .build(),
//!         )
//!         .blocking_rule(BlockingRule::on(&["surname"]))
//!         .build()?;
//!
//!     let mut linker = Linker::new(settings)?;
//!     let lf = df.lazy();
//!
//!     linker.estimate_probability_two_random_records_match(
//!         &lf,
//!         &[BlockingRule::on(&["first_name", "surname"])],
//!         1.0,
//!     )?;
//!     linker.estimate_u_using_random_sampling(&lf, 200)?;
//!     linker.estimate_parameters_using_em(&lf, &BlockingRule::on(&["surname"]))?;
//!
//!     let predictions = linker.predict(&lf, None)?.collect()?;
//!     let clusters = linker.cluster_pairwise_predictions(&predictions, 0.5)?;
//!     println!("{clusters}");
//!     Ok(())
//! }
//! ```

#[cfg(test)]
pub(crate) mod test_helpers;

pub mod blocking;
pub mod clustering;
pub mod comparison;
pub mod comparison_level;
pub mod comparison_vectors;
pub mod em;
pub mod error;
pub mod estimate_lambda;
pub mod estimate_u;
pub mod explain;
pub mod linker;
pub mod predict;
pub mod probability;
pub mod settings;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::blocking::BlockingRule;
    pub use crate::clustering::cluster_pairwise_predictions;
    pub use crate::comparison::Comparison;
    pub use crate::comparison_level::{ComparisonLevel, ComparisonPredicate};
    pub use crate::error::{Result, WeldrsError};
    pub use crate::explain::{
        ComparisonSummary, LevelSummary, ModelSummary, WaterfallChart, WaterfallStep,
    };
    pub use crate::linker::Linker;
    pub use crate::settings::{LinkType, Settings, TrainingSettings};
}
