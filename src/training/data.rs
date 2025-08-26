//! # Training Data
//!
//! Data structures and utilities for training data management.

use serde::{Deserialize, Serialize};

/// Type of problem for training
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProblemType {
    Math,
    Code,
    Logic,
    General,
}

/// Individual training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub input: String,
    pub target: String,
    pub reasoning_chain: Option<Vec<String>>,
    pub problem_type: ProblemType,
}

impl TrainingExample {
    /// Create a new training example
    pub fn new(input: String, target: String, problem_type: ProblemType) -> Self {
        Self {
            input,
            target,
            reasoning_chain: None,
            problem_type,
        }
    }

    /// Create a training example with reasoning chain
    pub fn with_reasoning(
        input: String,
        target: String,
        reasoning_chain: Vec<String>,
        problem_type: ProblemType,
    ) -> Self {
        Self {
            input,
            target,
            reasoning_chain: Some(reasoning_chain),
            problem_type,
        }
    }
}

/// Batch of training examples
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub examples: Vec<TrainingExample>,
    pub batch_size: usize,
}

impl TrainingBatch {
    /// Create a new training batch
    pub fn new(examples: Vec<TrainingExample>) -> Self {
        let batch_size = examples.len();
        Self {
            examples,
            batch_size,
        }
    }

    /// Split examples by problem type
    pub fn split_by_type(&self) -> std::collections::HashMap<ProblemType, Vec<&TrainingExample>> {
        let mut map = std::collections::HashMap::new();

        for example in &self.examples {
            map.entry(example.problem_type.clone())
                .or_insert_with(Vec::new)
                .push(example);
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_example_creation() {
        let example =
            TrainingExample::new("2 + 2 = ?".to_string(), "4".to_string(), ProblemType::Math);

        assert_eq!(example.input, "2 + 2 = ?");
        assert_eq!(example.target, "4");
        assert!(matches!(example.problem_type, ProblemType::Math));
        assert!(example.reasoning_chain.is_none());
    }

    #[test]
    fn test_training_example_with_reasoning() {
        let reasoning = vec!["I need to add 2 and 2".to_string(), "2 + 2 = 4".to_string()];

        let example = TrainingExample::with_reasoning(
            "2 + 2 = ?".to_string(),
            "4".to_string(),
            reasoning.clone(),
            ProblemType::Math,
        );

        assert_eq!(example.reasoning_chain, Some(reasoning));
    }

    #[test]
    fn test_training_batch() {
        let examples = vec![
            TrainingExample::new("2 + 2".to_string(), "4".to_string(), ProblemType::Math),
            TrainingExample::new("3 * 3".to_string(), "9".to_string(), ProblemType::Math),
        ];

        let batch = TrainingBatch::new(examples);
        assert_eq!(batch.batch_size, 2);

        let by_type = batch.split_by_type();
        assert_eq!(by_type.len(), 1);
        assert!(by_type.contains_key(&ProblemType::Math));
    }
}
