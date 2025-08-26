//! # Evaluation Utilities
//!
//! Model evaluation metrics and utilities.

use serde::{Deserialize, Serialize};
use crate::utils::error::{ModelError, Result};

/// Evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub accuracy: f32,
    pub reasoning_quality: f32,
    pub avg_reasoning_length: f32,
    pub examples_correct: usize,
    pub total_examples: usize,
    pub avg_generation_time_ms: f64,
}

impl EvalMetrics {
    /// Create new evaluation metrics
    pub fn new() -> Self {
        Self {
            accuracy: 0.0,
            reasoning_quality: 0.0,
            avg_reasoning_length: 0.0,
            examples_correct: 0,
            total_examples: 0,
            avg_generation_time_ms: 0.0,
        }
    }
    
    /// Update metrics with a new result
    pub fn update(&mut self, correct: bool, reasoning_length: usize, generation_time_ms: u64) {
        self.total_examples += 1;
        if correct {
            self.examples_correct += 1;
        }
        
        // Update running averages
        self.accuracy = self.examples_correct as f32 / self.total_examples as f32;
        
        let prev_avg_length = self.avg_reasoning_length;
        self.avg_reasoning_length = (prev_avg_length * (self.total_examples - 1) as f32 + reasoning_length as f32) / self.total_examples as f32;
        
        let prev_avg_time = self.avg_generation_time_ms;
        self.avg_generation_time_ms = (prev_avg_time * (self.total_examples - 1) as f64 + generation_time_ms as f64) / self.total_examples as f64;
    }
    
    /// Get formatted summary
    pub fn summary(&self) -> String {
        format!(
            "Accuracy: {:.2}% ({}/{}) | Avg Reasoning Length: {:.1} | Avg Time: {:.1}ms",
            self.accuracy * 100.0,
            self.examples_correct,
            self.total_examples,
            self.avg_reasoning_length,
            self.avg_generation_time_ms
        )
    }
}

impl Default for EvalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Difficulty levels for evaluation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// Evaluation example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalExample {
    pub problem: String,
    pub expected_answer: String,
    pub expected_reasoning: Option<Vec<String>>,
    pub difficulty: Difficulty,
    pub problem_type: String,
}

impl EvalExample {
    /// Create a new evaluation example
    pub fn new(
        problem: String,
        expected_answer: String,
        difficulty: Difficulty,
        problem_type: String,
    ) -> Self {
        Self {
            problem,
            expected_answer,
            expected_reasoning: None,
            difficulty,
            problem_type,
        }
    }
    
    /// Add expected reasoning chain
    pub fn with_reasoning(mut self, reasoning: Vec<String>) -> Self {
        self.expected_reasoning = Some(reasoning);
        self
    }
}

/// Model evaluator
pub struct Evaluator {
    // TODO: Add actual implementation in later tasks
}

impl Evaluator {
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {}
    }
    
    /// Evaluate model on a set of examples
    pub fn evaluate(&self, _examples: &[EvalExample]) -> Result<EvalMetrics> {
        // TODO: Implement in later tasks
        Err(ModelError::Evaluation("Evaluator not implemented yet".to_string()))
    }
    
    /// Check if answer is correct
    pub fn check_answer(&self, predicted: &str, expected: &str) -> bool {
        // Simple string matching for now
        predicted.trim().to_lowercase() == expected.trim().to_lowercase()
    }
    
    /// Evaluate reasoning quality (placeholder)
    pub fn evaluate_reasoning(&self, _predicted: &[String], _expected: &[String]) -> f32 {
        // TODO: Implement reasoning quality metrics
        0.5 // Placeholder score
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_metrics_creation() {
        let metrics = EvalMetrics::new();
        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.total_examples, 0);
        assert_eq!(metrics.examples_correct, 0);
    }

    #[test]
    fn test_eval_metrics_update() {
        let mut metrics = EvalMetrics::new();
        
        // Add correct answer
        metrics.update(true, 5, 100);
        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.total_examples, 1);
        assert_eq!(metrics.examples_correct, 1);
        
        // Add incorrect answer
        metrics.update(false, 3, 150);
        assert_eq!(metrics.accuracy, 0.5);
        assert_eq!(metrics.total_examples, 2);
        assert_eq!(metrics.examples_correct, 1);
    }

    #[test]
    fn test_eval_example_creation() {
        let example = EvalExample::new(
            "2 + 2 = ?".to_string(),
            "4".to_string(),
            Difficulty::Easy,
            "math".to_string(),
        );
        
        assert_eq!(example.problem, "2 + 2 = ?");
        assert_eq!(example.expected_answer, "4");
        assert_eq!(example.difficulty, Difficulty::Easy);
        assert!(example.expected_reasoning.is_none());
    }

    #[test]
    fn test_eval_example_with_reasoning() {
        let reasoning = vec!["Add 2 and 2".to_string(), "Result is 4".to_string()];
        let example = EvalExample::new(
            "2 + 2 = ?".to_string(),
            "4".to_string(),
            Difficulty::Easy,
            "math".to_string(),
        ).with_reasoning(reasoning.clone());
        
        assert_eq!(example.expected_reasoning, Some(reasoning));
    }

    #[test]
    fn test_evaluator_check_answer() {
        let evaluator = Evaluator::new();
        
        assert!(evaluator.check_answer("4", "4"));
        assert!(evaluator.check_answer("  4  ", "4"));
        assert!(evaluator.check_answer("Four", "four"));
        assert!(!evaluator.check_answer("4", "5"));
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics = EvalMetrics::new();
        metrics.update(true, 5, 100);
        metrics.update(false, 3, 150);
        
        let summary = metrics.summary();
        assert!(summary.contains("50.00%"));
        assert!(summary.contains("(1/2)"));
    }
}