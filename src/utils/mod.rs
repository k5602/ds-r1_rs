//! # Utilities Module
//!
//! Common utilities, error handling, and mathematical functions.

pub mod error;
pub mod math;
pub mod tokenizer;
pub mod evaluation;

// Re-export key types
pub use error::{ModelError, Result};
pub use math::MathUtils;
pub use tokenizer::{Tokenizer, TokenizerConfig};
pub use evaluation::{EvalMetrics, Evaluator};