//! # Training Module
//!
//! Training infrastructure including supervised learning and reinforcement learning.

pub mod trainer;
pub mod data;
pub mod optimizer;
pub mod loss;

// Re-export key types
pub use trainer::{BasicTrainer, RLTrainer};
pub use data::{TrainingExample, TrainingBatch, ProblemType};
pub use optimizer::{Optimizer, OptimizerConfig};
pub use loss::{LossFunction, TrainingMetrics};