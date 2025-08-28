//! # Evaluation Reports
//!
//! Comprehensive evaluation reporting system with visualization and comparison capabilities.

use crate::utils::error::{ModelError, Result};
use crate::utils::evaluation::{BenchmarkResults, DifficultyLevel, ProblemCategory, ProblemResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Comprehensive evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub report_id: String,
    pub timestamp: String,
    pub model_info: ModelInfo,
    pub overall_summary: OverallSummary,
    pub benchmark_results: Vec<BenchmarkResults>,
    pub comparative_analysis: Option<ComparativeAnalysis>,
    pub recommendations: Vec<String>,
}

/// Model information for the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_name: String,
    pub version: String,
    pub configuration: HashMap<String, String>,
    pub evaluation_date: String,
}

/// Overall summary across all benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallSummary {
    pub total_problems: usize,
    pub total_correct: usize,
    pub overall_accuracy: f32,
    pub avg_reasoning_quality: f32,
    pub total_evaluation_time: std::time::Duration,
    pub category_performance: HashMap<ProblemCategory, CategoryPerformance>,
    pub difficulty_performance: HashMap<DifficultyLevel, DifficultyPerformance>,
}

/// Performance metrics by category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryPerformance {
    pub accuracy: f32,
    pub avg_reasoning_quality: f32,
    pub problem_count: usize,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
}

/// Performance metrics by difficulty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyPerformance {
    pub accuracy: f32,
    pub avg_reasoning_quality: f32,
    pub problem_count: usize,
    pub avg_time: std::time::Duration,
}

/// Comparative analysis with baseline or previous results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    pub baseline_name: String,
    pub accuracy_improvement: f32,
    pub reasoning_quality_improvement: f32,
    pub performance_improvement: f32,
    pub category_improvements: HashMap<ProblemCategory, f32>,
    pub regression_areas: Vec<String>,
    pub improvement_areas: Vec<String>,
}

/// Evaluation report generator
pub struct EvaluationReportGenerator {
    baseline_results: Option<EvaluationReport>,
}

impl EvaluationReportGenerator {
    /// Create a new report generator
    pub fn new() -> Self {
        Self {
            baseline_results: None,
        }
    }

    /// Set baseline results for comparison
    pub fn with_baseline(mut self, baseline: EvaluationReport) -> Self {
        self.baseline_results = Some(baseline);
        self
    }

    /// Generate comprehensive evaluation report
    pub fn generate_report(
        &self,
        model_info: ModelInfo,
        benchmark_results: Vec<BenchmarkResults>,
    ) -> Result<EvaluationReport> {
        let report_id = format!("eval_{}", chrono::Utc::now().timestamp());
        let timestamp = chrono::Utc::now().to_rfc3339();

        let overall_summary = self.calculate_overall_summary(&benchmark_results);
        let comparative_analysis = self.generate_comparative_analysis(&overall_summary);
        let recommendations = self.generate_recommendations(&overall_summary, &benchmark_results);

        Ok(EvaluationReport {
            report_id,
            timestamp,
            model_info,
            overall_summary,
            benchmark_results,
            comparative_analysis,
            recommendations,
        })
    }

    /// Calculate overall summary across all benchmarks
    fn calculate_overall_summary(&self, results: &[BenchmarkResults]) -> OverallSummary {
        let total_problems: usize = results.iter().map(|r| r.total_problems).sum();
        let total_correct: usize = results.iter().map(|r| r.solved_correctly).sum();
        let overall_accuracy = if total_problems > 0 {
            total_correct as f32 / total_problems as f32
        } else {
            0.0
        };

        let total_evaluation_time = results
            .iter()
            .map(|r| r.performance_metrics.total_time)
            .fold(std::time::Duration::from_secs(0), |acc, time| acc + time);

        let avg_reasoning_quality = if !results.is_empty() {
            results
                .iter()
                .map(|r| r.average_metrics.overall_quality)
                .sum::<f32>()
                / results.len() as f32
        } else {
            0.0
        };

        let category_performance = self.calculate_category_performance(results);
        let difficulty_performance = self.calculate_difficulty_performance(results);

        OverallSummary {
            total_problems,
            total_correct,
            overall_accuracy,
            avg_reasoning_quality,
            total_evaluation_time,
            category_performance,
            difficulty_performance,
        }
    }

    /// Calculate performance by category
    fn calculate_category_performance(
        &self,
        results: &[BenchmarkResults],
    ) -> HashMap<ProblemCategory, CategoryPerformance> {
        let mut category_stats: HashMap<ProblemCategory, Vec<&ProblemResult>> = HashMap::new();

        // Collect all problem results by category
        for benchmark in results {
            for result in &benchmark.detailed_results {
                // Find the corresponding problem to get its category
                if let Some(category) = self.get_problem_category(benchmark, &result.problem_id) {
                    category_stats
                        .entry(category)
                        .or_insert_with(Vec::new)
                        .push(result);
                }
            }
        }

        category_stats
            .into_iter()
            .map(|(category, problem_results)| {
                let total = problem_results.len();
                let correct = problem_results.iter().filter(|r| r.is_correct).count();
                let accuracy = if total > 0 {
                    correct as f32 / total as f32
                } else {
                    0.0
                };

                let avg_reasoning_quality = if total > 0 {
                    problem_results
                        .iter()
                        .map(|r| r.reasoning_metrics.overall_quality)
                        .sum::<f32>()
                        / total as f32
                } else {
                    0.0
                };

                let (strengths, weaknesses) =
                    self.analyze_category_strengths_weaknesses(&problem_results);

                (
                    category,
                    CategoryPerformance {
                        accuracy,
                        avg_reasoning_quality,
                        problem_count: total,
                        strengths,
                        weaknesses,
                    },
                )
            })
            .collect()
    }

    /// Calculate performance by difficulty
    fn calculate_difficulty_performance(
        &self,
        results: &[BenchmarkResults],
    ) -> HashMap<DifficultyLevel, DifficultyPerformance> {
        let mut difficulty_stats: HashMap<DifficultyLevel, Vec<&ProblemResult>> = HashMap::new();

        // Collect all problem results by difficulty
        for benchmark in results {
            for result in &benchmark.detailed_results {
                if let Some(difficulty) = self.get_problem_difficulty(benchmark, &result.problem_id)
                {
                    difficulty_stats
                        .entry(difficulty)
                        .or_insert_with(Vec::new)
                        .push(result);
                }
            }
        }

        difficulty_stats
            .into_iter()
            .map(|(difficulty, problem_results)| {
                let total = problem_results.len();
                let correct = problem_results.iter().filter(|r| r.is_correct).count();
                let accuracy = if total > 0 {
                    correct as f32 / total as f32
                } else {
                    0.0
                };

                let avg_reasoning_quality = if total > 0 {
                    problem_results
                        .iter()
                        .map(|r| r.reasoning_metrics.overall_quality)
                        .sum::<f32>()
                        / total as f32
                } else {
                    0.0
                };

                let avg_time = if total > 0 {
                    let total_nanos: u128 = problem_results
                        .iter()
                        .map(|r| r.execution_time.as_nanos())
                        .sum();
                    std::time::Duration::from_nanos((total_nanos / total as u128) as u64)
                } else {
                    std::time::Duration::from_secs(0)
                };

                (
                    difficulty,
                    DifficultyPerformance {
                        accuracy,
                        avg_reasoning_quality,
                        problem_count: total,
                        avg_time,
                    },
                )
            })
            .collect()
    }

    /// Get problem category from benchmark (simplified lookup)
    fn get_problem_category(
        &self,
        _benchmark: &BenchmarkResults,
        problem_id: &str,
    ) -> Option<ProblemCategory> {
        // Simplified category detection based on problem ID prefix
        if problem_id.starts_with("math") {
            Some(ProblemCategory::Mathematics)
        } else if problem_id.starts_with("logic") {
            Some(ProblemCategory::Logic)
        } else if problem_id.starts_with("prog") || problem_id.starts_with("code") {
            Some(ProblemCategory::Programming)
        } else if problem_id.starts_with("sci") {
            Some(ProblemCategory::Science)
        } else {
            Some(ProblemCategory::General)
        }
    }

    /// Get problem difficulty from benchmark (simplified lookup)
    fn get_problem_difficulty(
        &self,
        _benchmark: &BenchmarkResults,
        problem_id: &str,
    ) -> Option<DifficultyLevel> {
        // Simplified difficulty detection - in a real implementation,
        // this would look up the actual problem definition
        if problem_id.contains("001") || problem_id.contains("002") || problem_id.contains("003") {
            Some(DifficultyLevel::Easy)
        } else if problem_id.contains("004")
            || problem_id.contains("005")
            || problem_id.contains("006")
        {
            Some(DifficultyLevel::Medium)
        } else {
            Some(DifficultyLevel::Hard)
        }
    }

    /// Analyze strengths and weaknesses for a category
    fn analyze_category_strengths_weaknesses(
        &self,
        results: &[&ProblemResult],
    ) -> (Vec<String>, Vec<String>) {
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        let accuracy =
            results.iter().filter(|r| r.is_correct).count() as f32 / results.len() as f32;
        let avg_reasoning_depth = results
            .iter()
            .map(|r| r.reasoning_metrics.reasoning_depth)
            .sum::<f32>()
            / results.len() as f32;
        let avg_step_clarity = results
            .iter()
            .map(|r| r.reasoning_metrics.step_clarity)
            .sum::<f32>()
            / results.len() as f32;

        if accuracy > 0.8 {
            strengths.push("High accuracy in problem solving".to_string());
        } else if accuracy < 0.5 {
            weaknesses.push("Low accuracy in problem solving".to_string());
        }

        if avg_reasoning_depth > 0.7 {
            strengths.push("Good reasoning depth and thoroughness".to_string());
        } else if avg_reasoning_depth < 0.4 {
            weaknesses.push("Insufficient reasoning depth".to_string());
        }

        if avg_step_clarity > 0.7 {
            strengths.push("Clear step-by-step reasoning".to_string());
        } else if avg_step_clarity < 0.4 {
            weaknesses.push("Unclear or missing step-by-step reasoning".to_string());
        }

        (strengths, weaknesses)
    }

    /// Generate comparative analysis with baseline
    fn generate_comparative_analysis(
        &self,
        summary: &OverallSummary,
    ) -> Option<ComparativeAnalysis> {
        if let Some(baseline) = &self.baseline_results {
            let accuracy_improvement =
                summary.overall_accuracy - baseline.overall_summary.overall_accuracy;
            let reasoning_quality_improvement =
                summary.avg_reasoning_quality - baseline.overall_summary.avg_reasoning_quality;

            let performance_improvement =
                if baseline.overall_summary.total_evaluation_time.as_secs_f32() > 0.0 {
                    (baseline.overall_summary.total_evaluation_time.as_secs_f32()
                        - summary.total_evaluation_time.as_secs_f32())
                        / baseline.overall_summary.total_evaluation_time.as_secs_f32()
                } else {
                    0.0
                };

            let mut category_improvements = HashMap::new();
            for (category, performance) in &summary.category_performance {
                if let Some(baseline_performance) =
                    baseline.overall_summary.category_performance.get(category)
                {
                    let improvement = performance.accuracy - baseline_performance.accuracy;
                    category_improvements.insert(category.clone(), improvement);
                }
            }

            let mut improvement_areas = Vec::new();
            let mut regression_areas = Vec::new();

            if accuracy_improvement > 0.05 {
                improvement_areas.push("Overall accuracy significantly improved".to_string());
            } else if accuracy_improvement < -0.05 {
                regression_areas.push("Overall accuracy decreased".to_string());
            }

            if reasoning_quality_improvement > 0.1 {
                improvement_areas.push("Reasoning quality improved".to_string());
            } else if reasoning_quality_improvement < -0.1 {
                regression_areas.push("Reasoning quality decreased".to_string());
            }

            Some(ComparativeAnalysis {
                baseline_name: baseline.model_info.model_name.clone(),
                accuracy_improvement,
                reasoning_quality_improvement,
                performance_improvement,
                category_improvements,
                regression_areas,
                improvement_areas,
            })
        } else {
            None
        }
    }

    /// Generate recommendations based on results
    fn generate_recommendations(
        &self,
        summary: &OverallSummary,
        _results: &[BenchmarkResults],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Overall accuracy recommendations
        if summary.overall_accuracy < 0.6 {
            recommendations.push(
                "Consider improving basic problem-solving accuracy through additional training"
                    .to_string(),
            );
        }

        // Reasoning quality recommendations
        if summary.avg_reasoning_quality < 0.5 {
            recommendations.push(
                "Focus on improving reasoning chain quality and step-by-step explanations"
                    .to_string(),
            );
        }

        // Category-specific recommendations
        for (category, performance) in &summary.category_performance {
            if performance.accuracy < 0.5 {
                match category {
                    ProblemCategory::Mathematics => {
                        recommendations.push(
                            "Strengthen mathematical reasoning and calculation skills".to_string(),
                        );
                    }
                    ProblemCategory::Logic => {
                        recommendations.push(
                            "Improve logical reasoning and deductive thinking capabilities"
                                .to_string(),
                        );
                    }
                    ProblemCategory::Programming => {
                        recommendations.push(
                            "Enhance code understanding and programming logic analysis".to_string(),
                        );
                    }
                    ProblemCategory::Science => {
                        recommendations.push(
                            "Expand scientific knowledge and reasoning capabilities".to_string(),
                        );
                    }
                    ProblemCategory::General => {
                        recommendations.push(
                            "Improve general reasoning and problem-solving skills".to_string(),
                        );
                    }
                }
            }
        }

        // Performance recommendations
        let avg_time_per_problem =
            summary.total_evaluation_time.as_secs_f32() / summary.total_problems as f32;
        if avg_time_per_problem > 10.0 {
            recommendations
                .push("Consider optimizing inference speed for better performance".to_string());
        }

        // Difficulty-specific recommendations
        if let Some(easy_perf) = summary.difficulty_performance.get(&DifficultyLevel::Easy) {
            if easy_perf.accuracy < 0.8 {
                recommendations.push(
                    "Focus on mastering basic concepts before tackling complex problems"
                        .to_string(),
                );
            }
        }

        if recommendations.is_empty() {
            recommendations.push(
                "Overall performance is satisfactory. Continue current training approach"
                    .to_string(),
            );
        }

        recommendations
    }

    /// Export report to JSON file
    pub fn export_to_json(&self, report: &EvaluationReport, file_path: &Path) -> Result<()> {
        let json_content = serde_json::to_string_pretty(report)
            .map_err(|e| ModelError::Evaluation(format!("Failed to serialize report: {}", e)))?;

        fs::write(file_path, json_content)
            .map_err(|e| ModelError::Evaluation(format!("Failed to write report file: {}", e)))?;

        Ok(())
    }

    /// Export report to markdown format
    pub fn export_to_markdown(&self, report: &EvaluationReport, file_path: &Path) -> Result<()> {
        let markdown_content = self.generate_markdown_report(report);

        fs::write(file_path, markdown_content).map_err(|e| {
            ModelError::Evaluation(format!("Failed to write markdown report: {}", e))
        })?;

        Ok(())
    }

    /// Generate markdown formatted report
    pub fn generate_markdown_report(&self, report: &EvaluationReport) -> String {
        let mut content = String::new();

        // Header
        content.push_str(&format!(
            "# Evaluation Report: {}\n\n",
            report.model_info.model_name
        ));
        content.push_str(&format!("**Report ID:** {}\n", report.report_id));
        content.push_str(&format!("**Generated:** {}\n", report.timestamp));
        content.push_str(&format!(
            "**Model Version:** {}\n\n",
            report.model_info.version
        ));

        // Overall Summary
        content.push_str("## Overall Summary\n\n");
        content.push_str(&format!(
            "- **Total Problems:** {}\n",
            report.overall_summary.total_problems
        ));
        content.push_str(&format!(
            "- **Correct Answers:** {}\n",
            report.overall_summary.total_correct
        ));
        content.push_str(&format!(
            "- **Overall Accuracy:** {:.1}%\n",
            report.overall_summary.overall_accuracy * 100.0
        ));
        content.push_str(&format!(
            "- **Average Reasoning Quality:** {:.2}\n",
            report.overall_summary.avg_reasoning_quality
        ));
        content.push_str(&format!(
            "- **Total Evaluation Time:** {:.2}s\n\n",
            report.overall_summary.total_evaluation_time.as_secs_f32()
        ));

        // Performance by Category
        content.push_str("## Performance by Category\n\n");
        for (category, performance) in &report.overall_summary.category_performance {
            content.push_str(&format!("### {:?}\n", category));
            content.push_str(&format!(
                "- **Accuracy:** {:.1}%\n",
                performance.accuracy * 100.0
            ));
            content.push_str(&format!(
                "- **Reasoning Quality:** {:.2}\n",
                performance.avg_reasoning_quality
            ));
            content.push_str(&format!("- **Problems:** {}\n", performance.problem_count));

            if !performance.strengths.is_empty() {
                content.push_str("- **Strengths:**\n");
                for strength in &performance.strengths {
                    content.push_str(&format!("  - {}\n", strength));
                }
            }

            if !performance.weaknesses.is_empty() {
                content.push_str("- **Weaknesses:**\n");
                for weakness in &performance.weaknesses {
                    content.push_str(&format!("  - {}\n", weakness));
                }
            }
            content.push('\n');
        }

        // Performance by Difficulty
        content.push_str("## Performance by Difficulty\n\n");
        for (difficulty, performance) in &report.overall_summary.difficulty_performance {
            content.push_str(&format!("### {:?}\n", difficulty));
            content.push_str(&format!(
                "- **Accuracy:** {:.1}%\n",
                performance.accuracy * 100.0
            ));
            content.push_str(&format!(
                "- **Reasoning Quality:** {:.2}\n",
                performance.avg_reasoning_quality
            ));
            content.push_str(&format!(
                "- **Average Time:** {:.2}s\n",
                performance.avg_time.as_secs_f32()
            ));
            content.push_str(&format!(
                "- **Problems:** {}\n\n",
                performance.problem_count
            ));
        }

        // Comparative Analysis
        if let Some(comparison) = &report.comparative_analysis {
            content.push_str("## Comparative Analysis\n\n");
            content.push_str(&format!("**Baseline:** {}\n\n", comparison.baseline_name));
            content.push_str(&format!(
                "- **Accuracy Change:** {:+.1}%\n",
                comparison.accuracy_improvement * 100.0
            ));
            content.push_str(&format!(
                "- **Reasoning Quality Change:** {:+.2}\n",
                comparison.reasoning_quality_improvement
            ));
            content.push_str(&format!(
                "- **Performance Change:** {:+.1}%\n\n",
                comparison.performance_improvement * 100.0
            ));

            if !comparison.improvement_areas.is_empty() {
                content.push_str("**Improvements:**\n");
                for improvement in &comparison.improvement_areas {
                    content.push_str(&format!("- {}\n", improvement));
                }
                content.push('\n');
            }

            if !comparison.regression_areas.is_empty() {
                content.push_str("**Regressions:**\n");
                for regression in &comparison.regression_areas {
                    content.push_str(&format!("- {}\n", regression));
                }
                content.push('\n');
            }
        }

        // Recommendations
        content.push_str("## Recommendations\n\n");
        for (i, recommendation) in report.recommendations.iter().enumerate() {
            content.push_str(&format!("{}. {}\n", i + 1, recommendation));
        }

        content
    }
}

impl Default for EvaluationReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::evaluation::{BenchmarkResults, ProblemResult, ReasoningMetrics};
    use std::time::Duration;

    fn create_sample_model_info() -> ModelInfo {
        ModelInfo {
            model_name: "DeepSeek-R1-Test".to_string(),
            version: "0.1.0".to_string(),
            configuration: HashMap::new(),
            evaluation_date: "2024-01-01".to_string(),
        }
    }

    fn create_sample_benchmark_results() -> Vec<BenchmarkResults> {
        vec![BenchmarkResults {
            benchmark_name: "Test Benchmark".to_string(),
            total_problems: 2,
            solved_correctly: 1,
            average_metrics: ReasoningMetrics::new(0.5, 0.6, 0.7, 0.8),
            performance_metrics: crate::utils::evaluation::PerformanceMetrics::new(
                Duration::from_secs(10),
                &[Duration::from_secs(5), Duration::from_secs(5)],
                64.0,
                100,
                50,
            ),
            category_breakdown: HashMap::new(),
            difficulty_breakdown: HashMap::new(),
            detailed_results: vec![
                ProblemResult {
                    problem_id: "math_001".to_string(),
                    is_correct: true,
                    reasoning_metrics: ReasoningMetrics::new(1.0, 0.8, 0.9, 0.7),
                    execution_time: Duration::from_secs(5),
                    generated_answer: "42".to_string(),
                    expected_answer: "42".to_string(),
                    reasoning_chain: vec!["Step 1".to_string()],
                    error_message: None,
                },
                ProblemResult {
                    problem_id: "logic_001".to_string(),
                    is_correct: false,
                    reasoning_metrics: ReasoningMetrics::new(0.0, 0.4, 0.5, 0.9),
                    execution_time: Duration::from_secs(5),
                    generated_answer: "Wrong".to_string(),
                    expected_answer: "Right".to_string(),
                    reasoning_chain: vec!["Step 1".to_string()],
                    error_message: None,
                },
            ],
        }]
    }

    #[test]
    fn test_report_generation() {
        let generator = EvaluationReportGenerator::new();
        let model_info = create_sample_model_info();
        let benchmark_results = create_sample_benchmark_results();

        let report = generator
            .generate_report(model_info, benchmark_results)
            .unwrap();

        assert_eq!(report.model_info.model_name, "DeepSeek-R1-Test");
        assert_eq!(report.overall_summary.total_problems, 2);
        assert_eq!(report.overall_summary.total_correct, 1);
        assert_eq!(report.overall_summary.overall_accuracy, 0.5);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_markdown_generation() {
        let generator = EvaluationReportGenerator::new();
        let model_info = create_sample_model_info();
        let benchmark_results = create_sample_benchmark_results();

        let report = generator
            .generate_report(model_info, benchmark_results)
            .unwrap();
        let markdown = generator.generate_markdown_report(&report);

        assert!(markdown.contains("# Evaluation Report"));
        assert!(markdown.contains("## Overall Summary"));
        assert!(markdown.contains("## Performance by Category"));
        assert!(markdown.contains("## Recommendations"));
    }

    #[test]
    fn test_category_performance_calculation() {
        let generator = EvaluationReportGenerator::new();
        let benchmark_results = create_sample_benchmark_results();
        let category_performance = generator.calculate_category_performance(&benchmark_results);

        assert!(!category_performance.is_empty());

        // Should have detected math and logic categories
        assert!(category_performance.contains_key(&ProblemCategory::Mathematics));
        assert!(category_performance.contains_key(&ProblemCategory::Logic));
    }

    #[test]
    fn test_recommendations_generation() {
        let generator = EvaluationReportGenerator::new();
        let model_info = create_sample_model_info();
        let benchmark_results = create_sample_benchmark_results();

        let report = generator
            .generate_report(model_info, benchmark_results)
            .unwrap();

        // Should generate recommendations for low accuracy
        assert!(!report.recommendations.is_empty());
        assert!(
            report
                .recommendations
                .iter()
                .any(|r| r.contains("accuracy"))
        );
    }
}
