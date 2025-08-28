//! # Training Module
//!
//! Training infrastructure including supervised learning and reinforcement learning.

pub mod code_dataset;
pub mod data;
pub mod loss;
pub mod math_dataset;
pub mod optimizer;
pub mod test_datasets;
pub mod trainer;

// Re-export key types
pub use code_dataset::{CodeExample, CodeExamplesDataset, AlgorithmType, CodeExplanationExpected};
pub use data::{DataLoader, ProblemType, SyntheticDataGenerator, TrainingBatch, TrainingExample};
pub use loss::{CrossEntropyLoss, LossFunction, TrainingMetrics};
pub use math_dataset::{MathDataset, MathProblem, DifficultyLevel, DatasetStatistics};
pub use optimizer::{Optimizer, OptimizerConfig};
pub use test_datasets::{
    TestDatasetCollection, create_extended_math_dataset, create_extended_logic_dataset,
    create_extended_programming_dataset, create_code_explanation_dataset,
    create_word_problems_dataset, create_multi_step_reasoning_dataset,
};
pub use trainer::{
    BasicTrainer, PolicyGradient, RLEvaluationMetrics, RLTrainer, RewardFunction,
    SimpleRewardFunction,
};
