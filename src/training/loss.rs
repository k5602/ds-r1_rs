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
    /// Small epsilon to prevent log(0)
    epsilon: f32,
}

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss
    pub fn new() -> Self {
        Self { epsilon: 1e-8 }
    }

    /// Compute softmax probabilities
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }

    /// Compute accuracy from predictions and targets
    pub fn compute_accuracy(&self, predictions: &[f32], targets: &[u32]) -> f32 {
        if predictions.is_empty() || targets.is_empty() {
            return 0.0;
        }

        let vocab_size = predictions.len() / targets.len();
        let mut correct = 0;

        for (i, &target) in targets.iter().enumerate() {
            let start_idx = i * vocab_size;
            let end_idx = start_idx + vocab_size;

            if end_idx <= predictions.len() {
                let logits = &predictions[start_idx..end_idx];
                let predicted_class = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if predicted_class == target as usize {
                    correct += 1;
                }
            }
        }

        correct as f32 / targets.len() as f32
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for CrossEntropyLoss {
    fn compute_loss(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        if predictions.is_empty() || targets.is_empty() {
            return Err(ModelError::Training(
                "Empty predictions or targets".to_string(),
            ));
        }

        // For cross-entropy loss, predictions should be [num_samples * vocab_size]
        // and targets should be [num_samples] with class indices
        let num_samples = targets.len();
        let vocab_size = predictions.len() / num_samples;

        if predictions.len() != num_samples * vocab_size {
            return Err(ModelError::Training(format!(
                "Predictions length {} doesn't match expected {} (targets: {} * vocab_size: {})",
                predictions.len(),
                num_samples * vocab_size,
                num_samples,
                vocab_size
            )));
        }

        let mut total_loss = 0.0;

        for i in 0..num_samples {
            let start_idx = i * vocab_size;
            let end_idx = start_idx + vocab_size;

            let logits = &predictions[start_idx..end_idx];
            let probs = self.softmax(logits);

            let target_idx = targets[i] as usize;
            if target_idx < probs.len() {
                let prob = probs[target_idx].max(self.epsilon);
                total_loss -= prob.ln();
            }
        }

        Ok(total_loss / num_samples as f32)
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

    #[test]
    fn test_softmax_computation() {
        let loss = CrossEntropyLoss::new();
        let logits = vec![1.0, 2.0, 3.0];
        let probs = loss.softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that probabilities are positive
        for prob in probs {
            assert!(prob > 0.0);
        }
    }

    #[test]
    fn test_accuracy_computation() {
        let loss = CrossEntropyLoss::new();

        // Perfect predictions
        let predictions = vec![
            10.0, 1.0, 1.0, // Class 0 (correct)
            1.0, 10.0, 1.0, // Class 1 (correct)
        ];
        let targets = vec![0, 1];

        let accuracy = loss.compute_accuracy(&predictions, &targets);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    fn test_cross_entropy_loss_computation() {
        let loss = CrossEntropyLoss::new();

        // Simple case with 2 classes
        let predictions = vec![1.0, 0.0, 0.0, 1.0]; // 2 samples, 2 classes each
        let targets = vec![0.0, 1.0]; // First sample class 0, second sample class 1

        let result = loss.compute_loss(&predictions, &targets);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_loss_with_empty_inputs() {
        let loss = CrossEntropyLoss::new();

        let result = loss.compute_loss(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_loss_with_mismatched_sizes() {
        let loss = CrossEntropyLoss::new();

        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![0.0, 1.0];

        let result = loss.compute_loss(&predictions, &targets);
        assert!(result.is_err());
    }
}
