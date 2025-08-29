//! # Model Evaluation Utilities
//!
//! Comprehensive evaluation framework for reasoning quality assessment and performance benchmarking.

use crate::inference::reasoning::{ReasoningAnalysis, StructuredReasoningOutput};
use crate::utils::error::{ModelError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

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

        // 1) Numeric tolerance (preferred for math-like answers)
        // Try to extract the first numeric value from both strings and compare with tolerance.
        let extract_number = |s: &str| -> Option<f64> {
            let bytes = s.as_bytes();
            let mut i = 0usize;
            while i < bytes.len() {
                let c = bytes[i] as char;
                // Potential start of a number
                if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' {
                    let mut j = i;
                    let mut has_digit = false;
                    while j < bytes.len() {
                        let cj = bytes[j] as char;
                        if cj.is_ascii_digit() {
                            has_digit = true;
                        }
                        if cj.is_ascii_digit()
                            || cj == '.'
                            || cj == 'e'
                            || cj == 'E'
                            || cj == '-'
                            || cj == '+'
                        {
                            j += 1;
                        } else {
                            break;
                        }
                    }
                    if has_digit {
                        if let Ok(val) = s[i..j].parse::<f64>() {
                            return Some(val);
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
            None
        };

        if let (Some(a), Some(b)) = (
            extract_number(&expected_clean),
            extract_number(&actual_clean),
        ) {
            let abs_diff = (a - b).abs();
            let abs_tol = 1e-4f64;
            let rel_tol = 1e-3f64;
            let tol = abs_tol.max(rel_tol * a.abs().max(b.abs()));
            if abs_diff <= tol {
                return 1.0;
            }
        }

        // 2) Exact or containment match
        if expected_clean == actual_clean {
            1.0
        } else if actual_clean.contains(&expected_clean) || expected_clean.contains(&actual_clean) {
            0.7
        } else {
            // 3) Simple word-overlap fallback
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

/// Performance metrics for timing and resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_time: Duration,
    pub avg_time_per_problem: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub memory_usage_mb: f64,
    pub tokens_per_second: f32,
    pub reasoning_overhead: f32, // Ratio of thinking tokens to total tokens
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new(
        total_time: Duration,
        problem_times: &[Duration],
        memory_usage_mb: f64,
        total_tokens: usize,
        thinking_tokens: usize,
    ) -> Self {
        let avg_time = if !problem_times.is_empty() {
            let total_nanos = problem_times.iter().map(|d| d.as_nanos()).sum::<u128>();
            let avg_nanos = total_nanos / problem_times.len() as u128;
            Duration::from_nanos(avg_nanos as u64)
        } else {
            Duration::from_secs(0)
        };

        let min_time = problem_times.iter().min().copied().unwrap_or_default();
        let max_time = problem_times.iter().max().copied().unwrap_or_default();

        let tokens_per_second = if total_time.as_secs_f32() > 0.0 {
            total_tokens as f32 / total_time.as_secs_f32()
        } else {
            0.0
        };

        let reasoning_overhead = if total_tokens > 0 {
            thinking_tokens as f32 / total_tokens as f32
        } else {
            0.0
        };

        Self {
            total_time,
            avg_time_per_problem: avg_time,
            min_time,
            max_time,
            memory_usage_mb,
            tokens_per_second,
            reasoning_overhead,
        }
    }
}

/// Evaluation results for a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub benchmark_name: String,
    pub total_problems: usize,
    pub solved_correctly: usize,
    pub average_metrics: ReasoningMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub category_breakdown: HashMap<ProblemCategory, CategoryResults>,
    pub difficulty_breakdown: HashMap<DifficultyLevel, DifficultyResults>,
    pub detailed_results: Vec<ProblemResult>,
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
    pub avg_time: Duration,
}

/// Individual problem evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemResult {
    pub problem_id: String,
    pub is_correct: bool,
    pub reasoning_metrics: ReasoningMetrics,
    pub execution_time: Duration,
    pub generated_answer: String,
    pub expected_answer: String,
    pub reasoning_chain: Vec<String>,
    pub error_message: Option<String>,
}

/// Evaluation harness for comprehensive testing
pub struct EvaluationHarness {
    evaluator: ReasoningEvaluator,
    enable_performance_tracking: bool,
    memory_tracking: bool,
    detailed_logging: bool,
}

impl EvaluationHarness {
    /// Create a new evaluation harness
    pub fn new() -> Self {
        Self {
            evaluator: ReasoningEvaluator::default(),
            enable_performance_tracking: true,
            memory_tracking: false, // Simplified for prototype
            detailed_logging: true,
        }
    }

    /// Configure performance tracking
    pub fn with_performance_tracking(mut self, enabled: bool) -> Self {
        self.enable_performance_tracking = enabled;
        self
    }

    /// Configure memory tracking
    pub fn with_memory_tracking(mut self, enabled: bool) -> Self {
        self.memory_tracking = enabled;
        self
    }

    /// Configure detailed logging
    pub fn with_detailed_logging(mut self, enabled: bool) -> Self {
        self.detailed_logging = enabled;
        self
    }

    /// Add a benchmark to the harness
    pub fn add_benchmark(&mut self, benchmark: ReasoningBenchmark) {
        self.evaluator.add_benchmark(benchmark);
    }

    /// Run comprehensive evaluation on a benchmark
    pub fn evaluate_comprehensive<F>(
        &self,
        benchmark_name: &str,
        mut inference_fn: F,
    ) -> Result<BenchmarkResults>
    where
        F: FnMut(&str) -> Result<StructuredReasoningOutput>,
    {
        let benchmark = self
            .evaluator
            .get_benchmarks()
            .iter()
            .find(|b| b.name == benchmark_name)
            .ok_or_else(|| {
                ModelError::Evaluation(format!("Benchmark '{}' not found", benchmark_name))
            })?;

        let mut detailed_results = Vec::new();
        let mut problem_times = Vec::new();
        let mut total_tokens = 0;
        let mut total_thinking_tokens = 0;
        let start_time = Instant::now();

        if self.detailed_logging {
            log::info!("Starting evaluation of benchmark: {}", benchmark_name);
            log::info!("Total problems: {}", benchmark.problems.len());
        }

        for (i, problem) in benchmark.problems.iter().enumerate() {
            if self.detailed_logging {
                log::debug!(
                    "Evaluating problem {}/{}: {}",
                    i + 1,
                    benchmark.problems.len(),
                    problem.id
                );
            }

            let problem_start = Instant::now();

            let result = match inference_fn(&problem.problem_text) {
                Ok(output) => {
                    let execution_time = problem_start.elapsed();
                    problem_times.push(execution_time);

                    // Count tokens (simplified estimation)
                    let answer_tokens = output
                        .reasoning_output
                        .final_answer
                        .split_whitespace()
                        .count();
                    let thinking_tokens: usize = output
                        .reasoning_output
                        .thinking_chain
                        .iter()
                        .map(|s| s.split_whitespace().count())
                        .sum();

                    total_tokens += answer_tokens + thinking_tokens;
                    total_thinking_tokens += thinking_tokens;

                    let metrics = ReasoningMetrics::from_analysis(
                        &output.analysis,
                        Some(&problem.expected_answer),
                        &output.reasoning_output.final_answer,
                    );

                    let is_correct = metrics.accuracy > 0.7;

                    ProblemResult {
                        problem_id: problem.id.clone(),
                        is_correct,
                        reasoning_metrics: metrics,
                        execution_time,
                        generated_answer: output.reasoning_output.final_answer.clone(),
                        expected_answer: problem.expected_answer.clone(),
                        reasoning_chain: output.reasoning_output.thinking_chain.clone(),
                        error_message: None,
                    }
                }
                Err(e) => {
                    let execution_time = problem_start.elapsed();
                    problem_times.push(execution_time);
                    let error_message = Some(e.to_string());

                    if self.detailed_logging {
                        log::warn!("Error evaluating problem {}: {}", problem.id, e);
                    }

                    ProblemResult {
                        problem_id: problem.id.clone(),
                        is_correct: false,
                        reasoning_metrics: ReasoningMetrics::new(0.0, 0.0, 0.0, 0.0),
                        execution_time,
                        generated_answer: String::new(),
                        expected_answer: problem.expected_answer.clone(),
                        reasoning_chain: Vec::new(),
                        error_message,
                    }
                }
            };

            detailed_results.push(result);
        }

        let total_time = start_time.elapsed();

        // Calculate performance metrics
        let memory_usage_mb = if self.memory_tracking {
            self.estimate_memory_usage()
        } else {
            0.0
        };

        let performance_metrics = PerformanceMetrics::new(
            total_time,
            &problem_times,
            memory_usage_mb,
            total_tokens,
            total_thinking_tokens,
        );

        // Calculate aggregate results
        let solved_correctly = detailed_results.iter().filter(|r| r.is_correct).count();

        let average_metrics = self.calculate_average_metrics(&detailed_results);
        let category_breakdown = self.calculate_category_breakdown(benchmark, &detailed_results);
        let difficulty_breakdown =
            self.calculate_difficulty_breakdown(benchmark, &detailed_results);

        if self.detailed_logging {
            log::info!(
                "Evaluation completed. Accuracy: {}/{} ({:.1}%)",
                solved_correctly,
                benchmark.problems.len(),
                (solved_correctly as f32 / benchmark.problems.len() as f32) * 100.0
            );
            log::info!(
                "Total time: {:?}, Avg time per problem: {:?}",
                total_time,
                performance_metrics.avg_time_per_problem
            );
        }

        Ok(BenchmarkResults {
            benchmark_name: benchmark_name.to_string(),
            total_problems: benchmark.problems.len(),
            solved_correctly,
            average_metrics,
            performance_metrics,
            category_breakdown,
            difficulty_breakdown,
            detailed_results,
        })
    }

    /// Calculate average metrics across all problems
    fn calculate_average_metrics(&self, results: &[ProblemResult]) -> ReasoningMetrics {
        if results.is_empty() {
            return ReasoningMetrics::new(0.0, 0.0, 0.0, 0.0);
        }

        let total = results.len() as f32;
        let avg_accuracy = results
            .iter()
            .map(|r| r.reasoning_metrics.accuracy)
            .sum::<f32>()
            / total;
        let avg_depth = results
            .iter()
            .map(|r| r.reasoning_metrics.reasoning_depth)
            .sum::<f32>()
            / total;
        let avg_clarity = results
            .iter()
            .map(|r| r.reasoning_metrics.step_clarity)
            .sum::<f32>()
            / total;
        let avg_verification = results
            .iter()
            .map(|r| r.reasoning_metrics.verification_presence)
            .sum::<f32>()
            / total;

        ReasoningMetrics::new(avg_accuracy, avg_depth, avg_clarity, avg_verification)
    }

    /// Calculate category breakdown
    fn calculate_category_breakdown(
        &self,
        benchmark: &ReasoningBenchmark,
        results: &[ProblemResult],
    ) -> HashMap<ProblemCategory, CategoryResults> {
        let mut category_stats: HashMap<ProblemCategory, Vec<&ProblemResult>> = HashMap::new();

        for (problem, result) in benchmark.problems.iter().zip(results.iter()) {
            category_stats
                .entry(problem.category.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        category_stats
            .into_iter()
            .map(|(category, results)| {
                let total = results.len();
                let correct = results.iter().filter(|r| r.is_correct).count();
                let accuracy = correct as f32 / total as f32;
                let avg_quality = results
                    .iter()
                    .map(|r| r.reasoning_metrics.overall_quality)
                    .sum::<f32>()
                    / total as f32;

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
            .collect()
    }

    /// Calculate difficulty breakdown
    fn calculate_difficulty_breakdown(
        &self,
        benchmark: &ReasoningBenchmark,
        results: &[ProblemResult],
    ) -> HashMap<DifficultyLevel, DifficultyResults> {
        let mut difficulty_stats: HashMap<DifficultyLevel, Vec<&ProblemResult>> = HashMap::new();

        for (problem, result) in benchmark.problems.iter().zip(results.iter()) {
            difficulty_stats
                .entry(problem.difficulty.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        difficulty_stats
            .into_iter()
            .map(|(difficulty, results)| {
                let total = results.len();
                let correct = results.iter().filter(|r| r.is_correct).count();
                let accuracy = correct as f32 / total as f32;
                let avg_quality = results
                    .iter()
                    .map(|r| r.reasoning_metrics.overall_quality)
                    .sum::<f32>()
                    / total as f32;
                let total_nanos = results
                    .iter()
                    .map(|r| r.execution_time.as_nanos())
                    .sum::<u128>();
                let avg_time = Duration::from_nanos((total_nanos / total as u128) as u64);

                (
                    difficulty,
                    DifficultyResults {
                        total,
                        correct,
                        accuracy,
                        avg_reasoning_quality: avg_quality,
                        avg_time,
                    },
                )
            })
            .collect()
    }

    /// Estimate memory usage (simplified for prototype)
    fn estimate_memory_usage(&self) -> f64 {
        // Simplified memory estimation - in a real implementation,
        // this would use system APIs to get actual memory usage
        64.0 // Placeholder: 64MB
    }

    /// Get available benchmarks
    pub fn get_benchmarks(&self) -> &[ReasoningBenchmark] {
        self.evaluator.get_benchmarks()
    }

    /// Run evaluation on all available benchmarks
    pub fn evaluate_all<F>(&self, mut inference_fn: F) -> Result<Vec<BenchmarkResults>>
    where
        F: FnMut(&str) -> Result<StructuredReasoningOutput>,
    {
        let benchmark_names: Vec<String> = self
            .get_benchmarks()
            .iter()
            .map(|b| b.name.clone())
            .collect();

        let mut all_results = Vec::new();
        for benchmark_name in benchmark_names {
            let result = self.evaluate_comprehensive(&benchmark_name, &mut inference_fn)?;
            all_results.push(result);
        }

        Ok(all_results)
    }

    /// Export a single benchmark's results to pretty JSON
    pub fn evaluate_comprehensive_json<F>(
        &self,
        benchmark_name: &str,
        mut inference_fn: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<StructuredReasoningOutput>,
    {
        let results = self.evaluate_comprehensive(benchmark_name, &mut inference_fn)?;
        serde_json::to_string_pretty(&results).map_err(ModelError::Json)
    }

    /// Evaluate all benchmarks and export as pretty JSON array
    pub fn evaluate_all_json<F>(&self, mut inference_fn: F) -> Result<String>
    where
        F: FnMut(&str) -> Result<StructuredReasoningOutput>,
    {
        let results = self.evaluate_all(&mut inference_fn)?;
        serde_json::to_string_pretty(&results).map_err(ModelError::Json)
    }
}

impl Default for EvaluationHarness {
    fn default() -> Self {
        Self::new()
    }
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
                        avg_time: Duration::from_secs(0), // Placeholder for old method
                    },
                )
            })
            .collect();

        Ok(BenchmarkResults {
            benchmark_name: benchmark_name.to_string(),
            total_problems: benchmark.problems.len(),
            solved_correctly,
            average_metrics,
            performance_metrics: PerformanceMetrics::new(Duration::from_secs(0), &[], 0.0, 0, 0), // Placeholder for old method
            category_breakdown,
            difficulty_breakdown,
            detailed_results: Vec::new(), // Placeholder for old method
        })
    }

    /// Create a comprehensive mathematics benchmark
    pub fn create_math_benchmark() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Mathematical Reasoning".to_string(),
            problems: vec![
                // Basic Arithmetic - Easy
                ReasoningProblem {
                    id: "math_001".to_string(),
                    problem_text: "What is 15% of 80?".to_string(),
                    expected_answer: "12".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_002".to_string(),
                    problem_text: "Calculate: 24 × 7 + 15".to_string(),
                    expected_answer: "183".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_003".to_string(),
                    problem_text: "If a rectangle has length 12m and width 8m, what is its area?".to_string(),
                    expected_answer: "96 square meters".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "math_004".to_string(),
                    problem_text: "What is the average of 15, 23, 18, and 32?".to_string(),
                    expected_answer: "22".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Easy,
                },

                // Algebra - Medium
                ReasoningProblem {
                    id: "math_005".to_string(),
                    problem_text: "Solve for x: 2x + 5 = 13".to_string(),
                    expected_answer: "4".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_006".to_string(),
                    problem_text: "If 3x - 7 = 2x + 8, what is the value of x?".to_string(),
                    expected_answer: "15".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_007".to_string(),
                    problem_text: "A number increased by 25% becomes 60. What was the original number?".to_string(),
                    expected_answer: "48".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },

                // Word Problems - Medium to Hard
                ReasoningProblem {
                    id: "math_008".to_string(),
                    problem_text: "Sarah has twice as many apples as John. Together they have 18 apples. How many apples does Sarah have?".to_string(),
                    expected_answer: "12".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_009".to_string(),
                    problem_text: "A train travels 240 km in 3 hours. At this rate, how long will it take to travel 400 km?".to_string(),
                    expected_answer: "5 hours".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "math_010".to_string(),
                    problem_text: "The sum of three consecutive integers is 48. What is the largest of these integers?".to_string(),
                    expected_answer: "17".to_string(),
                    category: ProblemCategory::Mathematics,
                    difficulty: DifficultyLevel::Hard,
                },
            ],
        }
    }

    /// Create a comprehensive logic benchmark
    pub fn create_logic_benchmark() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Logical Reasoning".to_string(),
            problems: vec![
                // Basic Logic - Easy
                ReasoningProblem {
                    id: "logic_001".to_string(),
                    problem_text: "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?".to_string(),
                    expected_answer: "Yes".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "logic_002".to_string(),
                    problem_text: "If all birds can fly, and a penguin is a bird, can a penguin fly?".to_string(),
                    expected_answer: "Yes, according to the premise".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "logic_003".to_string(),
                    problem_text: "Either it's sunny or it's raining. It's not sunny. What can we conclude?".to_string(),
                    expected_answer: "It's raining".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Easy,
                },

                // Conditional Logic - Medium
                ReasoningProblem {
                    id: "logic_004".to_string(),
                    problem_text: "If it rains, the ground gets wet. The ground is wet. Did it rain?".to_string(),
                    expected_answer: "Not necessarily".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "logic_005".to_string(),
                    problem_text: "If John studies hard, he will pass the exam. John passed the exam. Did John study hard?".to_string(),
                    expected_answer: "Not necessarily".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "logic_006".to_string(),
                    problem_text: "All roses are flowers. Some flowers are red. Are all roses red?".to_string(),
                    expected_answer: "No".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Medium,
                },

                // Complex Logic - Hard
                ReasoningProblem {
                    id: "logic_007".to_string(),
                    problem_text: "In a group of 5 people, everyone shakes hands with everyone else exactly once. How many handshakes occur?".to_string(),
                    expected_answer: "10".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Hard,
                },
                ReasoningProblem {
                    id: "logic_008".to_string(),
                    problem_text: "Three friends - Alice, Bob, and Carol - have different favorite colors: red, blue, and green. Alice doesn't like red. Bob doesn't like blue. Carol likes green. What color does Alice like?".to_string(),
                    expected_answer: "Blue".to_string(),
                    category: ProblemCategory::Logic,
                    difficulty: DifficultyLevel::Hard,
                },
            ],
        }
    }

    /// Get all available benchmarks
    pub fn get_benchmarks(&self) -> &[ReasoningBenchmark] {
        &self.benchmarks
    }

    /// Create a sample programming benchmark
    pub fn create_programming_benchmark() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Programming Logic".to_string(),
            problems: vec![
                ReasoningProblem {
                    id: "prog_001".to_string(),
                    problem_text: "What will this code output? for i in range(3): print(i)".to_string(),
                    expected_answer: "0\n1\n2".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "prog_002".to_string(),
                    problem_text: "Debug this function: def factorial(n): if n <= 1: return 1; return n * factorial(n-1)".to_string(),
                    expected_answer: "The function is correct".to_string(),
                    category: ProblemCategory::Programming,
                    difficulty: DifficultyLevel::Medium,
                },
            ],
        }
    }

    /// Create a sample science benchmark
    pub fn create_science_benchmark() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "Science Reasoning".to_string(),
            problems: vec![
                ReasoningProblem {
                    id: "sci_001".to_string(),
                    problem_text: "If water boils at 100°C at sea level, what happens to the boiling point at higher altitudes?".to_string(),
                    expected_answer: "It decreases".to_string(),
                    category: ProblemCategory::Science,
                    difficulty: DifficultyLevel::Medium,
                },
                ReasoningProblem {
                    id: "sci_002".to_string(),
                    problem_text: "What is the chemical formula for water?".to_string(),
                    expected_answer: "H2O".to_string(),
                    category: ProblemCategory::Science,
                    difficulty: DifficultyLevel::Easy,
                },
            ],
        }
    }

    /// Create a sample general reasoning benchmark
    pub fn create_general_benchmark() -> ReasoningBenchmark {
        ReasoningBenchmark {
            name: "General Reasoning".to_string(),
            problems: vec![
                ReasoningProblem {
                    id: "gen_001".to_string(),
                    problem_text: "If today is Monday, what day will it be in 10 days?".to_string(),
                    expected_answer: "Thursday".to_string(),
                    category: ProblemCategory::General,
                    difficulty: DifficultyLevel::Easy,
                },
                ReasoningProblem {
                    id: "gen_002".to_string(),
                    problem_text: "A farmer has 17 sheep. All but 9 die. How many are left?"
                        .to_string(),
                    expected_answer: "9".to_string(),
                    category: ProblemCategory::General,
                    difficulty: DifficultyLevel::Medium,
                },
            ],
        }
    }
}

impl Default for ReasoningEvaluator {
    fn default() -> Self {
        let mut evaluator = Self::new();
        evaluator.add_benchmark(Self::create_math_benchmark());
        evaluator.add_benchmark(Self::create_logic_benchmark());
        evaluator.add_benchmark(Self::create_programming_benchmark());
        evaluator.add_benchmark(Self::create_science_benchmark());
        evaluator.add_benchmark(Self::create_general_benchmark());
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
        assert_eq!(math_benchmark.name, "Mathematical Reasoning");
        assert!(!math_benchmark.problems.is_empty());

        let logic_benchmark = ReasoningEvaluator::create_logic_benchmark();
        assert_eq!(logic_benchmark.name, "Logical Reasoning");
        assert!(!logic_benchmark.problems.is_empty());
    }

    #[test]
    fn test_evaluator_default() {
        let evaluator = ReasoningEvaluator::default();
        assert_eq!(evaluator.get_benchmarks().len(), 5);
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

    #[test]
    fn test_numeric_tolerance_accuracy() {
        // Exact numeric with formatting differences
        assert_eq!(
            ReasoningMetrics::calculate_answer_accuracy("3.14", "3.1400"),
            1.0
        );

        // Within small absolute tolerance
        let acc = ReasoningMetrics::calculate_answer_accuracy("1000", "1000.00009");
        assert!(acc >= 1.0 - 1e-6);

        // Outside tolerance should not be full credit
        let acc2 = ReasoningMetrics::calculate_answer_accuracy("2.0", "2.5");
        assert!(acc2 < 1.0);
    }
}
