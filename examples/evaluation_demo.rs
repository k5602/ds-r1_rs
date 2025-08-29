//! # Evaluation System Demo
//!
//! Demonstrates the comprehensive evaluation and benchmarking system.

use ds_r1_rs::inference::reasoning::{ReasoningChainParser, StructuredReasoningOutput};
use ds_r1_rs::utils::error::Result;
use ds_r1_rs::utils::{
    DifficultyLevel, EvaluationHarness, EvaluationReportGenerator, ModelInfo, ProblemCategory,
    ReasoningBenchmark, ReasoningEvaluator, TestDatasets,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("🧠 DeepSeek R1 Evaluation System Demo");
    println!("=====================================\n");

    // Demo 1: Basic evaluation harness
    demo_evaluation_harness()?;

    // Demo 2: Comprehensive test datasets
    demo_test_datasets()?;

    // Demo 3: Evaluation reports
    demo_evaluation_reports()?;

    // Demo 4: Performance benchmarking
    demo_performance_benchmarking()?;

    println!("\n✅ Evaluation system demo completed successfully!");
    Ok(())
}

/// Demonstrate the evaluation harness functionality
fn demo_evaluation_harness() -> Result<()> {
    println!("📊 Demo 1: Evaluation Harness");
    println!("-----------------------------");

    // Create evaluation harness with performance tracking
    let mut harness = EvaluationHarness::new()
        .with_performance_tracking(true)
        .with_detailed_logging(true);

    // Add a simple benchmark
    let simple_benchmark = ReasoningBenchmark {
        name: "Demo Benchmark".to_string(),
        problems: vec![
            ds_r1_rs::utils::evaluation::ReasoningProblem {
                id: "demo_001".to_string(),
                problem_text: "What is 2 + 2?".to_string(),
                expected_answer: "4".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Easy,
            },
            ds_r1_rs::utils::evaluation::ReasoningProblem {
                id: "demo_002".to_string(),
                problem_text: "If all cats are mammals and Fluffy is a cat, is Fluffy a mammal?"
                    .to_string(),
                expected_answer: "Yes".to_string(),
                category: ProblemCategory::Logic,
                difficulty: DifficultyLevel::Easy,
            },
        ],
    };

    harness.add_benchmark(simple_benchmark);

    // Mock inference function for demonstration
    let mock_inference = |problem_text: &str| -> Result<StructuredReasoningOutput> {
        let mut parser = ReasoningChainParser::new(100, 101);

        let mock_response = match problem_text {
            text if text.contains("2 + 2") => {
                "<think>I need to add 2 and 2. That's 4.</think> The answer is 4."
            }
            text if text.contains("Fluffy") => {
                "<think>All cats are mammals. Fluffy is a cat. Therefore, Fluffy is a mammal.</think> Yes, Fluffy is a mammal."
            }
            _ => "<think>I'm not sure about this problem.</think> I don't know.",
        };

        parser.parse_structured(mock_response)
    };

    // Run evaluation
    let results = harness.evaluate_comprehensive("Demo Benchmark", mock_inference)?;

    println!("📈 Results:");
    println!("  - Total problems: {}", results.total_problems);
    println!("  - Correct answers: {}", results.solved_correctly);
    println!(
        "  - Accuracy: {:.1}%",
        (results.solved_correctly as f32 / results.total_problems as f32) * 100.0
    );
    println!(
        "  - Average reasoning quality: {:.2}",
        results.average_metrics.overall_quality
    );
    println!(
        "  - Total time: {:.2}s",
        results.performance_metrics.total_time.as_secs_f32()
    );
    println!();

    Ok(())
}

/// Demonstrate comprehensive test datasets
fn demo_test_datasets() -> Result<()> {
    println!("📚 Demo 2: Test Datasets");
    println!("------------------------");

    // Get all available datasets
    let datasets = TestDatasets::get_all_datasets();

    println!("Available datasets:");
    for dataset in &datasets {
        println!("  📖 {}", dataset.name);
        println!("     - Problems: {}", dataset.problems.len());

        // Count by difficulty
        let easy_count = dataset
            .problems
            .iter()
            .filter(|p| p.difficulty == DifficultyLevel::Easy)
            .count();
        let medium_count = dataset
            .problems
            .iter()
            .filter(|p| p.difficulty == DifficultyLevel::Medium)
            .count();
        let hard_count = dataset
            .problems
            .iter()
            .filter(|p| p.difficulty == DifficultyLevel::Hard)
            .count();

        println!(
            "     - Easy: {}, Medium: {}, Hard: {}",
            easy_count, medium_count, hard_count
        );

        // Show sample problem
        if let Some(sample) = dataset.problems.first() {
            println!(
                "     - Sample: {}",
                sample.problem_text.chars().take(50).collect::<String>()
            );
            if sample.problem_text.len() > 50 {
                println!("...");
            }
        }
        println!();
    }

    // Demonstrate reasoning chain validation
    let reasoning_examples = TestDatasets::create_reasoning_chain_dataset();
    println!("🔗 Reasoning Chain Validation:");

    for example in reasoning_examples.iter().take(2) {
        println!("  Problem: {}", example.problem);
        println!(
            "  Expected steps: {}",
            example.expected_reasoning_steps.len()
        );

        // Test validation with perfect match
        let perfect_score = example.validate_reasoning_chain(&example.expected_reasoning_steps);
        println!("  Perfect match score: {:.2}", perfect_score);

        // Test with partial match
        let partial_steps = vec![example.expected_reasoning_steps[0].clone()];
        let partial_score = example.validate_reasoning_chain(&partial_steps);
        println!("  Partial match score: {:.2}", partial_score);
        println!();
    }

    Ok(())
}

/// Demonstrate evaluation report generation
fn demo_evaluation_reports() -> Result<()> {
    println!("📋 Demo 3: Evaluation Reports");
    println!("-----------------------------");

    // Create mock model info
    let model_info = ModelInfo {
        model_name: "DeepSeek-R1-Demo".to_string(),
        version: "0.1.0".to_string(),
        configuration: {
            let mut config = HashMap::new();
            config.insert("hidden_size".to_string(), "512".to_string());
            config.insert("num_layers".to_string(), "8".to_string());
            config.insert("vocab_size".to_string(), "32000".to_string());
            config
        },
        evaluation_date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
    };

    // Create mock benchmark results
    let benchmark_results = vec![
        create_mock_benchmark_result("Mathematics", 10, 8, 0.8),
        create_mock_benchmark_result("Logic", 8, 6, 0.75),
        create_mock_benchmark_result("Programming", 12, 7, 0.58),
    ];

    // Generate report
    let report_generator = EvaluationReportGenerator::new();
    let report = report_generator.generate_report(model_info, benchmark_results)?;

    println!("📊 Generated Report:");
    println!("  - Report ID: {}", report.report_id);
    println!(
        "  - Model: {} v{}",
        report.model_info.model_name, report.model_info.version
    );
    println!(
        "  - Overall Accuracy: {:.1}%",
        report.overall_summary.overall_accuracy * 100.0
    );
    println!(
        "  - Reasoning Quality: {:.2}",
        report.overall_summary.avg_reasoning_quality
    );
    println!(
        "  - Total Problems: {}",
        report.overall_summary.total_problems
    );
    println!("  - Recommendations: {}", report.recommendations.len());

    // Show category breakdown
    println!("\n  📈 Category Performance:");
    for (category, performance) in &report.overall_summary.category_performance {
        println!(
            "    {:?}: {:.1}% accuracy, {:.2} quality",
            category,
            performance.accuracy * 100.0,
            performance.avg_reasoning_quality
        );
    }

    // Show recommendations
    println!("\n  💡 Recommendations:");
    for (i, recommendation) in report.recommendations.iter().enumerate() {
        println!("    {}. {}", i + 1, recommendation);
    }

    // Export to markdown (demo - would normally write to file)
    let markdown_content = report_generator.generate_markdown_report(&report);
    println!(
        "\n  📝 Markdown report generated ({} characters)",
        markdown_content.len()
    );

    Ok(())
}

/// Demonstrate performance benchmarking
fn demo_performance_benchmarking() -> Result<()> {
    println!("⚡ Demo 4: Performance Benchmarking");
    println!("----------------------------------");

    // Create evaluator with default benchmarks
    let evaluator = ReasoningEvaluator::default();

    println!("Available benchmarks:");
    for benchmark in evaluator.get_benchmarks() {
        println!(
            "  - {}: {} problems",
            benchmark.name,
            benchmark.problems.len()
        );
    }

    // Simulate performance metrics
    let performance_metrics = ds_r1_rs::utils::evaluation::PerformanceMetrics::new(
        std::time::Duration::from_secs(45),
        &[
            std::time::Duration::from_secs(2),
            std::time::Duration::from_secs(3),
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(4),
        ],
        128.0, // Memory usage in MB
        1500,  // Total tokens
        450,   // Thinking tokens
    );

    println!("\n⏱️  Performance Metrics:");
    println!(
        "  - Total time: {:.2}s",
        performance_metrics.total_time.as_secs_f32()
    );
    println!(
        "  - Average time per problem: {:.2}s",
        performance_metrics.avg_time_per_problem.as_secs_f32()
    );
    println!(
        "  - Min time: {:.2}s",
        performance_metrics.min_time.as_secs_f32()
    );
    println!(
        "  - Max time: {:.2}s",
        performance_metrics.max_time.as_secs_f32()
    );
    println!(
        "  - Memory usage: {:.1} MB",
        performance_metrics.memory_usage_mb
    );
    println!(
        "  - Tokens per second: {:.1}",
        performance_metrics.tokens_per_second
    );
    println!(
        "  - Reasoning overhead: {:.1}%",
        performance_metrics.reasoning_overhead * 100.0
    );

    Ok(())
}

/// Create mock benchmark result for demonstration
fn create_mock_benchmark_result(
    name: &str,
    total: usize,
    correct: usize,
    quality: f32,
) -> ds_r1_rs::utils::evaluation::BenchmarkResults {
    use ds_r1_rs::utils::evaluation::{
        BenchmarkResults, PerformanceMetrics, ProblemResult, ReasoningMetrics,
    };
    use std::time::Duration;

    let detailed_results: Vec<ProblemResult> = (0..total)
        .map(|i| ProblemResult {
            problem_id: format!("{}_{:03}", name.to_lowercase(), i + 1),
            is_correct: i < correct,
            reasoning_metrics: ReasoningMetrics::new(
                if i < correct { 1.0 } else { 0.0 },
                quality,
                quality * 0.9,
                quality * 0.8,
            ),
            execution_time: Duration::from_secs(2 + (i % 3) as u64),
            generated_answer: if i < correct {
                "Correct answer"
            } else {
                "Wrong answer"
            }
            .to_string(),
            expected_answer: "Expected answer".to_string(),
            reasoning_chain: vec![format!("Reasoning step for problem {}", i + 1)],
            error_message: None,
        })
        .collect();

    let performance_metrics = PerformanceMetrics::new(
        Duration::from_secs(total as u64 * 2),
        &vec![Duration::from_secs(2); total],
        64.0,
        total * 50,
        total * 15,
    );

    BenchmarkResults {
        benchmark_name: name.to_string(),
        total_problems: total,
        solved_correctly: correct,
        average_metrics: ReasoningMetrics::new(
            correct as f32 / total as f32,
            quality,
            quality * 0.9,
            quality * 0.8,
        ),
        performance_metrics,
        category_breakdown: std::collections::HashMap::new(),
        difficulty_breakdown: std::collections::HashMap::new(),
        detailed_results,
    }
}
