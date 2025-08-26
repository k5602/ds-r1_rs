//! # Training Module
//!
//! Training infrastructure including supervised learning and reinforcement learning.

pub mod data;
pub mod loss;
pub mod optimizer;
pub mod trainer;

// Re-export key types
pub use data::{ProblemType, TrainingBatch, TrainingExample};
pub use loss::{LossFunction, TrainingMetrics};
pub use optimizer::{Optimizer, OptimizerConfig};
pub use trainer::{BasicTrainer, RLTrainer};
