//! # Utilities Module
//!
//! Common utilities, error handling, and mathematical functions.

pub mod error;
pub mod evaluation;
pub mod math;
pub mod tokenizer;

// Re-export key types
pub use error::{ModelError, Result};
pub use evaluation::{EvalMetrics, Evaluator};
pub use math::MathUtils;
pub use tokenizer::{Tokenizer, TokenizerConfig};
