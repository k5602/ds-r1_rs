//! # Loss Functions
//!
//! Loss function implementations for training.

use crate::utils::error::{ModelError, Result};

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub perplexity: f32,
    pub step: usize,
}

impl TrainingMetrics {
    /// Create new training metrics
    pub fn new(loss: f32, accuracy: f32, step: usize) -> Self {
        let perplexity = loss.exp();
        Self {
            loss,
            accuracy,
            perplexity,
            step,
        }
    }
}

/// Loss function trait
pub trait LossFunction {
    fn compute_loss(&self, predictions: &[f32], targets: &[f32]) -> Result<f32>;
}

/// Cross-entropy loss for language modeling
pub struct CrossEntropyLoss {
    // TODO: Add actual implementation in later tasks
}

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for CrossEntropyLoss {
    fn compute_loss(&self, _predictions: &[f32], _targets: &[f32]) -> Result<f32> {
        // TODO: Implement in later tasks
        Err(ModelError::Training(
            "Cross-entropy loss not implemented yet".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics() {
        let metrics = TrainingMetrics::new(2.0, 0.8, 100);
        assert_eq!(metrics.loss, 2.0);
        assert_eq!(metrics.accuracy, 0.8);
        assert_eq!(metrics.step, 100);
        assert!((metrics.perplexity - 2.0_f32.exp()).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss_creation() {
        let _loss = CrossEntropyLoss::new();
        let _loss_default = CrossEntropyLoss::default();
    }
}
