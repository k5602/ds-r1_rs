//! # Utilities Module
//!
//! Common utilities, error handling, and mathematical functions.

pub mod error;
pub mod evaluation;
pub mod evaluation_reports;
pub mod math;
pub mod test_datasets;
pub mod tokenizer;

// Re-export key types
pub use error::{ModelError, Result};
pub use evaluation::{
    ReasoningEvaluator, ReasoningMetrics, ReasoningBenchmark, EvaluationHarness,
    PerformanceMetrics, BenchmarkResults, ProblemResult, ProblemCategory, DifficultyLevel
};
pub use evaluation_reports::{
    EvaluationReport, EvaluationReportGenerator, ModelInfo, OverallSummary,
    CategoryPerformance, DifficultyPerformance, ComparativeAnalysis
};
pub use math::MathUtils;
pub use test_datasets::{TestDatasets, ReasoningChainExample};
pub use tokenizer::{Tokenizer, TokenizerConfig};
