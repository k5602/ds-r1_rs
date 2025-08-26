//! # Model Evaluation Utilities
//!
//! Utilities for evaluating model performance and reasoning quality.

use crate::inference::reasoning::{ReasoningAnalysis, StructuredReasoningOutput};
use crate::utils::error::{ModelError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evaluation metrics for reasoning quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningMetrics {
    pub accuracy: f32,
    pub reasoning_depth: f32,
    pub step_clarity: f32,
    pub verification_presence: f32,
    pub overall_quality: f32,
}

impl ReasoningMetrics {
    /// Create new reasoning metrics
    pub fn new(
        accuracy: f32,
        reasoning_depth: f32,
        step_clarity: f32,
        verification_presence: f32,
    ) -> Self {
        let overall_quality = (accuracy * 0.4
            + reasoning_depth * 0.3
            + step_clarity * 0.2
            + verification_presence * 0.1)
            .min(1.0);

        Self {
            accuracy,
            reasoning_depth,
            step_clarity,
            verification_presence,
            overall_quality,
        }
    }

    /// Calculate metrics from reasoning analysis
    pub fn from_analysis(
        analysis: &ReasoningAnalysis,
        expected_answer: Option<&str>,
        actual_answer: &str,
    ) -> Self {
        let accuracy = if let Some(expected) = expected_answer {
            Self::calculate_answer_accuracy(expected, actual_answer)
        } else {
            1.0 // No ground truth available
        };

        let reasoning_depth = (analysis.total_thinking_sections as f32 * 0.2).min(1.0);
        let step_clarity = if analysis.has_step_by_step { 0.8 } else { 0.3 };
        let verification_presence = if analysis.has_verification { 1.0 } else { 0.0 };

        Self::new(
            accuracy,
            reasoning_depth,
            step_clarity,
            verification_presence,
        )
    }

    /// Calculate accuracy between expected and actual answers
    fn calculate_answer_accuracy(expected: &str, actual: &str) -> f32 {
        let expected_clean = expected.trim().to_lowercase();
        let actual_clean = actual.trim().to_lowercase();

        if expected_clean == actual_clean {
            1.0
        } else if actual_clean.contains(&expected_clean) || expected_clean.contains(&actual_clean) {
            0.7
        } else {
            // Simple word overlap metric
            let expected_words: std::collections::HashSet<&str> =
                expected_clean.split_whitespace().collect();
            let actual_words: std::collections::HashSet<&str> =
                actual_clean.split_whitespace().collect();

            let intersection = expected_words.intersection(&actual_words).count();
            let union = expected_words.union(&actual_words).count();

            if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            }
        }
    }
}

/// Evaluation benchmark for reasoning tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningBenchmark {
    pub name: String,
    pub problems: Vec<ReasoningProblem>,
}

/// Individual reasoning problem for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningProblem {
    pub id: String,
    pub problem_text: String,
    pub expected_answer: String,
    pub category: ProblemCategory,
    pub difficulty: DifficultyLevel,
}

/// Categories of reasoning problems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub enum ProblemCategory {
    Mathematics,
    Logic,
    Programming,
    Science,
    General,
}

/// Difficulty levels for problems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
}

/// Evaluation results for a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub benchmark_name: String,
    pub total_problems: usize,
    pub solved_correctly: usize,
    pub average_metrics: ReasoningMetrics,
    pub category_breakdown: HashMap<ProblemCategory, CategoryResults>,
    pub difficulty_breakdown: HashMap<DifficultyLevel, DifficultyResults>,
}

/// Results for a specific category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryResults {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub avg_reasoning_quality: f32,
}

/// Results for a specific difficulty level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyResults {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub avg_reasoning_quality: f32,
}

/// Reasoning evaluator
pub struct ReasoningEvaluator {
    benchmarks: Vec<ReasoningBenchmark>,
}

impl ReasoningEvaluator {
    /// Create a new reasoning evaluator
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    /// Add a benchmark to the evaluator
    pub fn add_benchmark(&mut self, benchmark: ReasoningBenchmark) {
        self.benchmarks.push(benchmark);
    }

    /// Evaluate reasoning outputs against a benchmark
    pub fn evaluate_benchmark(
        &self,
        benchmark_name: &str,
        reasoning_outputs: &[StructuredReasoningOutput],
    ) -> Result<BenchmarkResults> {
        let benchmark = self
            .benchmarks
            .iter()
            .find(|b| b.name == benchmark_name)
            .ok_or_else(|| {
                ModelError::Evaluation(format!("Benchmark '{}' not found", benchmark_name))
            })?;

        if reasoning_outputs.len() != benchmark.problems.len() {
            return Err(ModelError::Evaluation(
                "Number of outputs doesn't match number of problems".to_string(),
            ));
        }

        let mut total_metrics = Vec::new();
        let mut category_stats: HashMap<ProblemCategory, Vec<(bool, f32)>> = HashMap::new();
        let mut difficulty_stats: HashMap<DifficultyLevel, Vec<(bool, f32)>> = HashMap::new();
        let mut solved_correctly = 0;

        for (problem, output) in benchmark.problems.iter().zip(reasoning_outputs.iter()) {
            let metrics = ReasoningMetrics::from_analysis(
                &output.analysis,
                Some(&problem.expected_answer),
                &output.reasoning_output.final_answer,
            );

            let is_correct = metrics.accuracy > 0.7; // Threshold for "correct"
            if is_correct {
                solved_correctly += 1;
            }

            total_metrics.push(metrics.clone());

            // Category stats
            category_stats
                .entry(problem.category.clone())
                .or_insert_with(Vec::new)
                .push((is_correct, metrics.overall_quality));

            // Difficulty stats
            difficulty_stats
                .entry(problem.difficulty.clone())
                .or_insert_with(Vec::new)
                .push((is_correct, metrics.overall_quality));
        }

        // Calculate average metrics
        let avg_accuracy =
            total_metrics.iter().map(|m| m.accuracy).sum::<f32>() / total_metrics.len() as f32;
        let avg_depth = total_metrics.iter().map(|m| m.reasoning_depth).sum::<f32>()
            / total_metrics.len() as f32;
        let avg_clarity =
            total_metrics.iter().map(|m| m.step_clarity).sum::<f32>() / total_metrics.len() as f32;
        let avg_verification = total_metrics
            .iter()
            .map(|m| m.verification_presence)
            .sum::<f32>()
            / total_metrics.len() as f32;

        let average_metrics =
            ReasoningMetrics::new(avg_accuracy, avg_depth, avg_clarity, avg_verification);

        // Calculate category breakdown
        let category_breakdown = category_stats
            .into_iter()
            .map(|(category, results)| {
                let total = results.len();
                let correct = results.iter().filter(|(is_correct, _)| *is_correct).count();
                let accuracy = correct as f32 / total as f32;
                let avg_quality =
                    results.iter().map(|(_, quality)| *quality).sum::<f32>() / total as f32;

                (
                    category,
                    CategoryResults {
                        total,
                        correct,
                        accuracy,
                        avg_reasoning_quality: avg_quality,
                    },
                )
            })
            .collect();

        // Calculate difficulty breakdown
        let difficulty_breakdown = difficulty_stats
            .into_iter()
            .map(|(difficulty, results)| {
                let total = results.len();
                let correct = results.iter().filter(|(is_correct, _)| *is_correct).count();
                let accuracy = correct as f32 / total as f32;
                let avg_quality =
                    results.iter().map(|(_, quality)| *quality).sum::<f32>() / total as f32;

                (
                    difficulty,
                    DifficultyResults {
                        total,
                        correct,
                        accuracy,
                        avg_reasoning_quality: avg_quality,
                    },
                )
            })
            .collect();

        Ok(BenchmarkResults {
            benchmark_name: benchmark_name.to_string(),
            total_problems: benchmark.problems.len(),
            solved_correctly,
            average_metrics,
            category_breakdown,
            difficulty_breakdown,
        })
    }

    /// Create a sample mathematics benchmark
    pub fn create_math_benchmark() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Basic Mathematics".to_string(),
            problems: vec![
                ReasoningProblem {
                    id: "math_001".to_string(),
                    problem_text: "What is 15% of 80?".to_string(),
                    expected_answer: "12".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_002".to_string(),
                    problem_text: "If a rectangle has length 12m and width 8m, what is its area?"
                        .to_string(),
                    expected_answer: "96 square meters".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_003".to_string(),
                    problem_text: "Solve for x: 2x + 5 = 13".to_string(),
                    expected_answer: "4".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
            ],
        }
    }

    /// Create a sample logic benchmark
    pub fn create_logic_benchmark() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Logical Reasoning".to_string(),
            problems: vec![
                ReasoningProblem {
                    id: "logic_001".to_string(),
                    problem_text: "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?"
                        .to_string(),
                    expected_answer: "Yes".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "logic_002".to_string(),
                    problem_text:
                        "If it rains, the ground gets wet. The ground is wet. Did it rain?"
                            .to_string(),
                    expected_answer: "Not necessarily".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Medium,
                },
            ],
        }
    }

    /// Get all available benchmarks
    pub fn get_benchmarks(&self) -> &[ReasoningBenchmark] {
        &self.benchmarks
    }
}

impl Default for ReasoningEvaluator {
    fn default() -> Self {
        let mut evaluator = Self::new();
        evaluator.add_benchmark(Self::create_math_benchmark());
        evaluator.add_benchmark(Self::create_logic_benchmark());
        evaluator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::reasoning::ReasoningAnalysis;

    #[test]
    fn test_reasoning_metrics_creation() {
        let metrics = ReasoningMetrics::new(0.9, 0.8, 0.7, 0.6);
        assert_eq!(metrics.accuracy, 0.9);
        assert_eq!(metrics.reasoning_depth, 0.8);
        assert_eq!(metrics.step_clarity, 0.7);
        assert_eq!(metrics.verification_presence, 0.6);
        assert!(metrics.overall_quality > 0.0);
        assert!(metrics.overall_quality <= 1.0);
    }

    #[test]
    fn test_answer_accuracy_calculation() {
        assert_eq!(ReasoningMetrics::calculate_answer_accuracy("42", "42"), 1.0);
        assert_eq!(
            ReasoningMetrics::calculate_answer_accuracy("42", "The answer is 42"),
            0.7
        );
        assert!(
            (ReasoningMetrics::calculate_answer_accuracy("hello world", "goodbye world")
                - 0.33333334)
                .abs()
                < 1e-6
        );
        assert_eq!(
            ReasoningMetrics::calculate_answer_accuracy("cat", "dog"),
            0.0
        );
    }

    #[test]
    fn test_metrics_from_analysis() {
        let thinking_chain = vec![
            "First, I need to understand the problem".to_string(),
            "Let me verify this calculation".to_string(),
        ];
        let analysis = ReasoningAnalysis::new(&thinking_chain);

        let metrics = ReasoningMetrics::from_analysis(&analysis, Some("42"), "42");
        assert_eq!(metrics.accuracy, 1.0);
        assert!(metrics.verification_presence > 0.0);
    }

    #[test]
    fn test_benchmark_creation() {
        let math_benchmark = ReasoningEvaluator::create_math_benchmark();
        assert_eq!(math_benchmark.name, "Basic Mathematics");
        assert!(!math_benchmark.problems.is_empty());

        let logic_benchmark = ReasoningEvaluator::create_logic_benchmark();
        assert_eq!(logic_benchmark.name, "Logical Reasoning");
        assert!(!logic_benchmark.problems.is_empty());
    }

    #[test]
    fn test_evaluator_default() {
        let evaluator = ReasoningEvaluator::default();
        assert_eq!(evaluator.get_benchmarks().len(), 2);
    }

    #[test]
    fn test_problem_categories() {
        let problem = ReasoningProblem {
            id: "test_001".to_string(),
            problem_text: "Test problem".to_string(),
            expected_answer: "Test answer".to_string(),
            category: ProblemCategory::Mathematics,
            difficulty: DifficultyLevel::Easy,
        };

        assert_eq!(problem.category, ProblemCategory::Mathematics);
        assert_eq!(problem.difficulty, DifficultyLevel::Easy);
    }
}
