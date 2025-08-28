//! # Code Reasoning Demonstration
//!
//! Demonstrates the code analysis and explanation capabilities with step-by-step
//! reasoning, algorithm analysis, and code understanding evaluation.

use ds_r1_rs::{
    inference::{CodeAnalyzer, CodeAnalysis, CodeExplanation},
    training::{CodeExamplesDataset, AlgorithmType},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Evaluation metrics for code understanding quality
#[derive(Debug, Clone)]
pub struct CodeUnderstandingMetrics {
    pub explanation_completeness: f32,    // 0.0 to 1.0
    pub technical_accuracy: f32,          // 0.0 to 1.0
    pub clarity_score: f32,               // 0.0 to 1.0
    pub complexity_analysis_quality: f32, // 0.0 to 1.0
    pub overall_score: f32,               // 0.0 to 1.0
}

/// Code reasoning demonstration system
pub struct CodeReasoningDemo {
    analyzer: CodeAnalyzer,
    dataset: CodeExamplesDataset,
}

impl CodeReasoningDemo {
    /// Create a new code reasoning demonstration
    pub fn new() -> Self {
        Self {
            analyzer: CodeAnalyzer::new(),
            dataset: CodeExamplesDataset::new(),
        }
    }

    /// Run comprehensive code reasoning demonstration
    pub fn run_demonstration(&self) -> Result<()> {
        println!("🔍 Code Understanding and Explanation Demonstration");
        println!("{}", "=".repeat(60));
        println!();

        // Demonstrate analysis of different algorithm types
        self.demonstrate_algorithm_analysis()?;
        
        // Show step-by-step code explanation
        self.demonstrate_step_by_step_explanation()?;
        
        // Demonstrate code generation with reasoning
        self.demonstrate_code_generation_reasoning()?;
        
        // Evaluate code understanding quality
        self.demonstrate_evaluation_metrics()?;

        println!("\n✅ Code reasoning demonstration completed successfully!");
        Ok(())
    }

    /// Demonstrate analysis of different algorithm types
    fn demonstrate_algorithm_analysis(&self) -> Result<()> {
        println!("📊 Algorithm Analysis Demonstration");
        println!("{}", "-".repeat(40));

        let algorithm_types = vec![
            AlgorithmType::Searching,
            AlgorithmType::Sorting,
            AlgorithmType::DataStructure,
        ];

        for algo_type in algorithm_types {
            let examples = self.dataset.get_examples_by_type(algo_type.clone());
            if let Some(example) = examples.first() {
                println!("\n🔍 Analyzing: {} ({:?})", example.name, algo_type);
                
                let analysis = self.analyzer.analyze_code(&example.code, &example.language)?;
                
                println!("  📈 Complexity Analysis:");
                println!("    • Time: {}", analysis.time_complexity);
                println!("    • Space: {}", analysis.space_complexity);
                
                println!("  🎯 Identified Patterns:");
                for pattern in &analysis.patterns {
                    println!("    • {:?}", pattern);
                }
                
                println!("  ⭐ Code Quality:");
                println!("    • Readability: {:.1}/1.0", analysis.quality.readability_score);
                println!("    • Maintainability: {:.1}/1.0", analysis.quality.maintainability_score);
                println!("    • Efficiency: {:.1}/1.0", analysis.quality.efficiency_score);
            }
        }

        Ok(())
    }

    /// Demonstrate step-by-step code explanation
    fn demonstrate_step_by_step_explanation(&self) -> Result<()> {
        println!("\n\n📝 Step-by-Step Code Explanation");
        println!("{}", "-".repeat(40));

        // Use binary search as an example
        let examples = self.dataset.get_examples_by_type(AlgorithmType::Searching);
        if let Some(binary_search_example) = examples.iter().find(|e| e.name == "Binary Search") {
            println!("\n🎯 Explaining: {}", binary_search_example.name);
            println!("\n📋 Code:");
            println!("{}", binary_search_example.code);
            
            let explanation = self.analyzer.explain_code(
                &binary_search_example.code, 
                &binary_search_example.language
            )?;
            
            println!("\n🔍 Overview:");
            println!("{}", explanation.overview);
            
            println!("\n📚 Step-by-Step Breakdown:");
            for (i, step) in explanation.steps.iter().enumerate() {
                println!("  {}. Code: `{}`", i + 1, step.code_section.trim());
                println!("     Explanation: {}", step.explanation);
                println!("     Reasoning: {}", step.reasoning);
                if !step.key_concepts.is_empty() {
                    println!("     Key Concepts: {}", step.key_concepts.join(", "));
                }
                println!();
            }
            
            println!("🧮 Complexity Analysis:");
            println!("{}", explanation.complexity_analysis);
            
            println!("\n🎨 Pattern Analysis:");
            println!("{}", explanation.pattern_analysis);
            
            if !explanation.best_practices_notes.is_empty() {
                println!("\n✅ Best Practices:");
                for practice in &explanation.best_practices_notes {
                    println!("  • {}", practice);
                }
            }
        }

        Ok(())
    }

    /// Demonstrate code generation with approach reasoning
    fn demonstrate_code_generation_reasoning(&self) -> Result<()> {
        println!("\n\n🛠️  Code Generation with Reasoning");
        println!("{}", "-".repeat(40));

        // Simulate reasoning about implementing a simple algorithm
        println!("\n💭 Problem: Implement a function to find the maximum element in an array");
        
        println!("\n🤔 Reasoning Process:");
        println!("1. **Approach Selection**: Linear scan is most straightforward");
        println!("   • Could use sorting (O(n log n)) but that's overkill");
        println!("   • Linear scan is O(n) and optimal for this problem");
        println!("   • Need to handle empty array edge case");
        
        println!("\n2. **Implementation Strategy**:");
        println!("   • Use iterator for idiomatic Rust code");
        println!("   • Return Option<T> to handle empty arrays safely");
        println!("   • Make it generic to work with any comparable type");
        
        println!("\n3. **Generated Code with Reasoning**:");
        let generated_code = r#"fn find_max<T: PartialOrd + Copy>(arr: &[T]) -> Option<T> {
    // Handle empty array case - return None for safety
    if arr.is_empty() {
        return None;
    }
    
    // Start with first element as initial maximum
    let mut max = arr[0];
    
    // Iterate through remaining elements
    for &element in arr.iter().skip(1) {
        // Update max if current element is larger
        if element > max {
            max = element;
        }
    }
    
    Some(max)
}"#;
        
        println!("{}", generated_code);
        
        println!("\n🎯 Reasoning Behind Design Choices:");
        println!("  • **Generic Type T**: Works with any comparable type (i32, f64, etc.)");
        println!("  • **PartialOrd Bound**: Enables comparison operations");
        println!("  • **Copy Bound**: Allows copying values without ownership issues");
        println!("  • **Option Return**: Safe handling of empty input");
        println!("  • **Iterator with skip(1)**: Avoids redundant self-comparison");
        
        // Analyze the generated code
        let analysis = self.analyzer.analyze_code(generated_code, "rust")?;
        println!("\n📊 Analysis of Generated Code:");
        println!("  • Time Complexity: {}", analysis.time_complexity);
        println!("  • Space Complexity: {}", analysis.space_complexity);
        println!("  • Quality Score: {:.1}/1.0", analysis.quality.readability_score);

        Ok(())
    }

    /// Demonstrate evaluation metrics for code understanding quality
    fn demonstrate_evaluation_metrics(&self) -> Result<()> {
        println!("\n\n📈 Code Understanding Quality Evaluation");
        println!("{}", "-".repeat(40));

        let mut total_metrics = Vec::new();

        // Evaluate understanding quality for different examples
        let examples = self.dataset.get_examples();
        for example in examples.iter().take(3) { // Evaluate first 3 examples
            println!("\n🔍 Evaluating: {}", example.name);
            
            let analysis = self.analyzer.analyze_code(&example.code, &example.language)?;
            let explanation = self.analyzer.explain_code(&example.code, &example.language)?;
            
            let metrics = self.evaluate_understanding_quality(&analysis, &explanation, example);
            
            println!("  📊 Quality Metrics:");
            println!("    • Explanation Completeness: {:.2}/1.0", metrics.explanation_completeness);
            println!("    • Technical Accuracy: {:.2}/1.0", metrics.technical_accuracy);
            println!("    • Clarity Score: {:.2}/1.0", metrics.clarity_score);
            println!("    • Complexity Analysis: {:.2}/1.0", metrics.complexity_analysis_quality);
            println!("    • Overall Score: {:.2}/1.0", metrics.overall_score);
            
            total_metrics.push(metrics);
        }

        // Calculate aggregate metrics
        let avg_metrics = self.calculate_average_metrics(&total_metrics);
        println!("\n🎯 Overall Performance Summary:");
        println!("  📊 Average Metrics Across All Examples:");
        println!("    • Explanation Completeness: {:.2}/1.0", avg_metrics.explanation_completeness);
        println!("    • Technical Accuracy: {:.2}/1.0", avg_metrics.technical_accuracy);
        println!("    • Clarity Score: {:.2}/1.0", avg_metrics.clarity_score);
        println!("    • Complexity Analysis: {:.2}/1.0", avg_metrics.complexity_analysis_quality);
        println!("    • **Overall Score: {:.2}/1.0**", avg_metrics.overall_score);

        // Provide performance interpretation
        self.interpret_performance(&avg_metrics);

        Ok(())
    }

    /// Evaluate the quality of code understanding and explanation
    fn evaluate_understanding_quality(
        &self,
        analysis: &CodeAnalysis,
        explanation: &CodeExplanation,
        expected: &ds_r1_rs::training::CodeExample,
    ) -> CodeUnderstandingMetrics {
        // Evaluate explanation completeness
        let explanation_completeness = self.evaluate_completeness(explanation, expected);
        
        // Evaluate technical accuracy
        let technical_accuracy = self.evaluate_technical_accuracy(analysis, expected);
        
        // Evaluate clarity
        let clarity_score = self.evaluate_clarity(explanation);
        
        // Evaluate complexity analysis quality
        let complexity_analysis_quality = self.evaluate_complexity_analysis(analysis, expected);
        
        // Calculate overall score
        let overall_score = (explanation_completeness + technical_accuracy + 
                           clarity_score + complexity_analysis_quality) / 4.0;

        CodeUnderstandingMetrics {
            explanation_completeness,
            technical_accuracy,
            clarity_score,
            complexity_analysis_quality,
            overall_score,
        }
    }

    /// Evaluate how complete the explanation is
    fn evaluate_completeness(
        &self,
        explanation: &CodeExplanation,
        expected: &ds_r1_rs::training::CodeExample,
    ) -> f32 {
        let expected_steps = expected.expected_explanation.step_by_step.len();
        let actual_steps = explanation.steps.len();
        
        // Score based on step coverage and content quality
        let step_coverage = (actual_steps.min(expected_steps) as f32) / (expected_steps as f32);
        let content_quality = if explanation.overview.len() > 50 { 0.8 } else { 0.5 };
        
        (step_coverage + content_quality) / 2.0
    }

    /// Evaluate technical accuracy of the analysis
    fn evaluate_technical_accuracy(
        &self,
        analysis: &CodeAnalysis,
        expected: &ds_r1_rs::training::CodeExample,
    ) -> f32 {
        let mut accuracy_score = 0.0;
        let mut total_checks = 0.0;

        // Check if key concepts are identified
        for concept in &expected.key_concepts {
            total_checks += 1.0;
            if analysis.explanation_steps.iter().any(|step| 
                step.to_lowercase().contains(&concept.to_lowercase())
            ) {
                accuracy_score += 1.0;
            }
        }

        // Check pattern identification
        total_checks += 1.0;
        if !analysis.patterns.is_empty() {
            accuracy_score += 0.8; // Partial credit for identifying any patterns
        }

        if total_checks > 0.0 {
            accuracy_score / total_checks
        } else {
            0.5 // Default score if no checks available
        }
    }

    /// Evaluate clarity of explanation
    fn evaluate_clarity(&self, explanation: &CodeExplanation) -> f32 {
        let mut clarity_score = 0.0;
        
        // Check overview quality
        if explanation.overview.len() > 30 && explanation.overview.len() < 200 {
            clarity_score += 0.3;
        }
        
        // Check step explanations
        if explanation.steps.len() >= 3 {
            clarity_score += 0.3;
        }
        
        // Check if complexity analysis is provided
        if !explanation.complexity_analysis.is_empty() {
            clarity_score += 0.2;
        }
        
        // Check if pattern analysis is provided
        if !explanation.pattern_analysis.is_empty() {
            clarity_score += 0.2;
        }
        
        clarity_score
    }

    /// Evaluate quality of complexity analysis
    fn evaluate_complexity_analysis(
        &self,
        analysis: &CodeAnalysis,
        _expected: &ds_r1_rs::training::CodeExample,
    ) -> f32 {
        let mut score = 0.0;
        
        // Check if time complexity is identified
        if analysis.time_complexity != ds_r1_rs::inference::Complexity::Unknown {
            score += 0.5;
        }
        
        // Check if space complexity is identified
        if analysis.space_complexity != ds_r1_rs::inference::Complexity::Unknown {
            score += 0.5;
        }
        
        score
    }

    /// Calculate average metrics across all evaluations
    fn calculate_average_metrics(&self, metrics: &[CodeUnderstandingMetrics]) -> CodeUnderstandingMetrics {
        if metrics.is_empty() {
            return CodeUnderstandingMetrics {
                explanation_completeness: 0.0,
                technical_accuracy: 0.0,
                clarity_score: 0.0,
                complexity_analysis_quality: 0.0,
                overall_score: 0.0,
            };
        }

        let count = metrics.len() as f32;
        CodeUnderstandingMetrics {
            explanation_completeness: metrics.iter().map(|m| m.explanation_completeness).sum::<f32>() / count,
            technical_accuracy: metrics.iter().map(|m| m.technical_accuracy).sum::<f32>() / count,
            clarity_score: metrics.iter().map(|m| m.clarity_score).sum::<f32>() / count,
            complexity_analysis_quality: metrics.iter().map(|m| m.complexity_analysis_quality).sum::<f32>() / count,
            overall_score: metrics.iter().map(|m| m.overall_score).sum::<f32>() / count,
        }
    }

    /// Interpret and provide feedback on performance
    fn interpret_performance(&self, metrics: &CodeUnderstandingMetrics) {
        println!("\n🎭 Performance Interpretation:");
        
        match metrics.overall_score {
            score if score >= 0.8 => {
                println!("  🌟 Excellent: The code understanding system demonstrates strong");
                println!("     analytical capabilities with comprehensive explanations.");
            }
            score if score >= 0.6 => {
                println!("  ✅ Good: The system provides solid code analysis with room for");
                println!("     improvement in explanation depth and technical accuracy.");
            }
            score if score >= 0.4 => {
                println!("  ⚠️  Fair: Basic code understanding is present but explanations");
                println!("     need more detail and technical precision.");
            }
            _ => {
                println!("  ❌ Needs Improvement: The system requires significant enhancement");
                println!("     in code analysis and explanation capabilities.");
            }
        }

        println!("\n💡 Recommendations:");
        if metrics.explanation_completeness < 0.7 {
            println!("  • Enhance explanation completeness with more detailed step-by-step breakdowns");
        }
        if metrics.technical_accuracy < 0.7 {
            println!("  • Improve technical accuracy by better pattern and concept identification");
        }
        if metrics.clarity_score < 0.7 {
            println!("  • Increase clarity through better structured explanations and examples");
        }
        if metrics.complexity_analysis_quality < 0.7 {
            println!("  • Strengthen complexity analysis with more precise algorithmic assessment");
        }
    }
}

impl Default for CodeReasoningDemo {
    fn default() -> Self {
        Self::new()
    }
}

fn main() -> Result<()> {
    let demo = CodeReasoningDemo::new();
    demo.run_demonstration()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_creation() {
        let demo = CodeReasoningDemo::new();
        assert!(!demo.dataset.is_empty());
    }

    #[test]
    fn test_metrics_calculation() {
        let demo = CodeReasoningDemo::new();
        let metrics = vec![
            CodeUnderstandingMetrics {
                explanation_completeness: 0.8,
                technical_accuracy: 0.7,
                clarity_score: 0.9,
                complexity_analysis_quality: 0.6,
                overall_score: 0.75,
            },
            CodeUnderstandingMetrics {
                explanation_completeness: 0.6,
                technical_accuracy: 0.8,
                clarity_score: 0.7,
                complexity_analysis_quality: 0.8,
                overall_score: 0.725,
            },
        ];

        let avg = demo.calculate_average_metrics(&metrics);
        assert!((avg.explanation_completeness - 0.7).abs() < 0.01);
        assert!((avg.technical_accuracy - 0.75).abs() < 0.01);
    }
}