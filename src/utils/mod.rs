//! # Utilities Module
//!
//! Common utilities, error handling, and mathematical functions.

pub mod checkpoint;
pub mod error;
pub mod evaluation;
pub mod evaluation_reports;
pub mod math;
pub mod test_datasets;
pub mod tokenizer;

// Re-export key types
pub use checkpoint::{load_weights_json, save_weights_json};
pub use error::{ModelError, Result};
pub use evaluation::{
    BenchmarkResults, DifficultyLevel, EvaluationHarness, PerformanceMetrics, ProblemCategory,
    ProblemResult, ReasoningBenchmark, ReasoningEvaluator, ReasoningMetrics,
};
pub use evaluation_reports::{
    CategoryPerformance, ComparativeAnalysis, DifficultyPerformance, EvaluationReport,
    EvaluationReportGenerator, ModelInfo, OverallSummary,
};
pub use math::MathUtils;
pub use test_datasets::{ReasoningChainExample, TestDatasets};
pub use tokenizer::{Tokenizer, TokenizerConfig};
