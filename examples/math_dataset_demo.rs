//! # Mathematical Dataset Demo
//!
//! Demonstrates the mathematical problem dataset for training and evaluation.

use ds_r1_rs::{
    inference::math_solver::{MathProblemSolver, MathProblemType},
    model::{config::ModelConfig},
    training::{MathDataset, DifficultyLevel},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DeepSeek R1 Mathematical Dataset Demo");
    println!("====================================\n");

    // Create the default mathematical dataset
    let dataset = MathDataset::create_default_dataset();
    
    // Display dataset statistics
    display_dataset_statistics(&dataset);
    
    // Show examples by problem type
    demonstrate_problem_types(&dataset);
    
    // Show examples by difficulty level
    demonstrate_difficulty_levels(&dataset);
    
    // Test problem solving with dataset
    test_problem_solving_with_dataset(&dataset)?;
    
    // Export and import dataset
    demonstrate_dataset_serialization(&dataset)?;

    println!("✅ Mathematical dataset demo completed!");
    
    Ok(())
}

/// Display comprehensive dataset statistics
fn display_dataset_statistics(dataset: &MathDataset) {
    println!("📊 Dataset Overview");
    println!("==================");
    println!("Name: {}", dataset.metadata.name);
    println!("Version: {}", dataset.metadata.version);
    println!("Description: {}", dataset.metadata.description);
    println!();
    
    let stats = dataset.get_statistics();
    println!("{}", stats.format_statistics());
}

/// Demonstrate different problem types in the dataset
fn demonstrate_problem_types(dataset: &MathDataset) {
    println!("🔢 Problem Types Examples");
    println!("========================\n");

    // Arithmetic problems
    println!("1. Arithmetic Problems:");
    println!("-----------------------");
    let arithmetic_problems = dataset.get_problems_by_type(&MathProblemType::Arithmetic);
    for (i, problem) in arithmetic_problems.iter().take(3).enumerate() {
        println!("Example {}: {}", i + 1, problem.problem_text);
        println!("Expected Answer: {}", problem.expected_answer);
        println!("Steps:");
        for (j, step) in problem.expected_steps.iter().enumerate() {
            println!("  {}. {}", j + 1, step);
        }
        println!();
    }

    // Algebra problems
    println!("2. Algebra Problems:");
    println!("--------------------");
    let algebra_problems = dataset.get_problems_by_type(&MathProblemType::Algebra);
    for (i, problem) in algebra_problems.iter().take(3).enumerate() {
        println!("Example {}: {}", i + 1, problem.problem_text);
        println!("Expected Answer: {}", problem.expected_answer);
        println!("Steps:");
        for (j, step) in problem.expected_steps.iter().enumerate() {
            println!("  {}. {}", j + 1, step);
        }
        println!();
    }

    // Word problems
    println!("3. Word Problems:");
    println!("-----------------");
    let word_problems = dataset.get_problems_by_type(&MathProblemType::WordProblem);
    for (i, problem) in word_problems.iter().take(3).enumerate() {
        println!("Example {}: {}", i + 1, problem.problem_text);
        println!("Expected Answer: {}", problem.expected_answer);
        println!("Steps:");
        for (j, step) in problem.expected_steps.iter().enumerate() {
            println!("  {}. {}", j + 1, step);
        }
        println!();
    }

    // Equation problems
    println!("4. Equation Problems:");
    println!("---------------------");
    let equation_problems = dataset.get_problems_by_type(&MathProblemType::Equation);
    for (i, problem) in equation_problems.iter().take(3).enumerate() {
        println!("Example {}: {}", i + 1, problem.problem_text);
        println!("Expected Answer: {}", problem.expected_answer);
        println!("Steps:");
        for (j, step) in problem.expected_steps.iter().enumerate() {
            println!("  {}. {}", j + 1, step);
        }
        println!();
    }
}

/// Demonstrate different difficulty levels
fn demonstrate_difficulty_levels(dataset: &MathDataset) {
    println!("📈 Difficulty Levels");
    println!("===================\n");

    // Elementary problems
    println!("1. Elementary Level:");
    println!("--------------------");
    let elementary_problems = dataset.get_problems_by_difficulty(&DifficultyLevel::Elementary);
    for (i, problem) in elementary_problems.iter().take(2).enumerate() {
        println!("Example {}: {}", i + 1, problem.problem_text);
        println!("Type: {:?}", problem.problem_type);
        println!("Answer: {}", problem.expected_answer);
        println!();
    }

    // Intermediate problems
    println!("2. Intermediate Level:");
    println!("----------------------");
    let intermediate_problems = dataset.get_problems_by_difficulty(&DifficultyLevel::Intermediate);
    for (i, problem) in intermediate_problems.iter().take(2).enumerate() {
        println!("Example {}: {}", i + 1, problem.problem_text);
        println!("Type: {:?}", problem.problem_type);
        println!("Answer: {}", problem.expected_answer);
        println!();
    }

    // Advanced problems (if any)
    let advanced_problems = dataset.get_problems_by_difficulty(&DifficultyLevel::Advanced);
    if !advanced_problems.is_empty() {
        println!("3. Advanced Level:");
        println!("------------------");
        for (i, problem) in advanced_problems.iter().take(2).enumerate() {
            println!("Example {}: {}", i + 1, problem.problem_text);
            println!("Type: {:?}", problem.problem_type);
            println!("Answer: {}", problem.expected_answer);
            println!();
        }
    }
}

/// Test problem solving with dataset problems
fn test_problem_solving_with_dataset(dataset: &MathDataset) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Testing Problem Solving");
    println!("=========================\n");

    let config = ModelConfig::default();
    let mut solver = MathProblemSolver::new(
        config.thinking_token_id,
        config.thinking_token_id + 1,
    );

    // Test with a few problems from each type
    let arithmetic_problems = dataset.get_problems_by_type(&MathProblemType::Arithmetic);
    let algebra_problems = dataset.get_problems_by_type(&MathProblemType::Algebra);
    let word_problems = dataset.get_problems_by_type(&MathProblemType::WordProblem);
    let equation_problems = dataset.get_problems_by_type(&MathProblemType::Equation);
    
    let test_problems = [
        arithmetic_problems.get(0),
        algebra_problems.get(0),
        word_problems.get(0),
        equation_problems.get(0),
    ];

    for (i, problem_opt) in test_problems.iter().enumerate() {
        if let Some(problem) = problem_opt {
            println!("Test {}: {}", i + 1, problem.problem_text);
            println!("Expected: {}", problem.expected_answer);
            
            match solver.solve_problem(&problem.problem_text) {
                Ok(solution) => {
                    println!("Solver Result: {:?}", solution.final_answer);
                    
                    // Check if the answer matches (simple string comparison)
                    let matches = solution.final_answer.as_ref()
                        .map(|ans| ans.contains(&problem.expected_answer) || problem.expected_answer.contains(ans))
                        .unwrap_or(false);
                    
                    println!("Match: {}", if matches { "✅" } else { "❌" });
                }
                Err(e) => println!("Error: {}", e),
            }
            println!();
        }
    }

    Ok(())
}

/// Demonstrate dataset serialization and deserialization
fn demonstrate_dataset_serialization(dataset: &MathDataset) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Dataset Serialization");
    println!("=======================\n");

    // Export to JSON
    println!("Exporting dataset to JSON...");
    let json_data = dataset.to_json()?;
    println!("JSON size: {} bytes", json_data.len());
    
    // Show a small sample of the JSON
    let sample = if json_data.len() > 200 {
        format!("{}...", &json_data[..200])
    } else {
        json_data.clone()
    };
    println!("JSON sample:\n{}\n", sample);

    // Import from JSON
    println!("Importing dataset from JSON...");
    let imported_dataset = MathDataset::from_json(&json_data)?;
    println!("Successfully imported {} problems", imported_dataset.problems.len());
    
    // Verify the import
    let original_stats = dataset.get_statistics();
    let imported_stats = imported_dataset.get_statistics();
    
    println!("Verification:");
    println!("- Original problems: {}", original_stats.total_problems);
    println!("- Imported problems: {}", imported_stats.total_problems);
    println!("- Match: {}", if original_stats.total_problems == imported_stats.total_problems { "✅" } else { "❌" });
    
    Ok(())
}

/// Demonstrate random sampling from dataset
fn demonstrate_random_sampling(dataset: &MathDataset) {
    println!("🎲 Random Sampling");
    println!("=================\n");

    println!("Random sample of 5 problems:");
    let sample = dataset.get_random_sample(5);
    
    for (i, problem) in sample.iter().enumerate() {
        println!("{}. [{}] {}", 
                 i + 1, 
                 format!("{:?}", problem.problem_type), 
                 problem.problem_text);
        println!("   Answer: {}", problem.expected_answer);
    }
    println!();
}

/// Demonstrate filtering problems by multiple criteria
fn demonstrate_advanced_filtering(dataset: &MathDataset) {
    println!("🔍 Advanced Filtering");
    println!("====================\n");

    // Get elementary arithmetic problems
    println!("Elementary Arithmetic Problems:");
    let filtered = dataset.get_problems_matching(
        Some(&MathProblemType::Arithmetic), 
        Some(&DifficultyLevel::Elementary)
    );
    println!("Found {} problems", filtered.len());
    for problem in filtered.iter().take(2) {
        println!("- {}", problem.problem_text);
    }
    println!();

    // Get intermediate algebra problems
    println!("Intermediate Algebra Problems:");
    let filtered = dataset.get_problems_matching(
        Some(&MathProblemType::Algebra), 
        Some(&DifficultyLevel::Intermediate)
    );
    println!("Found {} problems", filtered.len());
    for problem in filtered.iter().take(2) {
        println!("- {}", problem.problem_text);
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_demo_components() {
        let dataset = MathDataset::create_default_dataset();
        assert!(dataset.problems.len() > 0);
        
        let stats = dataset.get_statistics();
        assert!(stats.total_problems > 0);
    }

    #[test]
    fn test_problem_type_filtering() {
        let dataset = MathDataset::create_default_dataset();
        
        let arithmetic = dataset.get_problems_by_type(&MathProblemType::Arithmetic);
        let algebra = dataset.get_problems_by_type(&MathProblemType::Algebra);
        let word = dataset.get_problems_by_type(&MathProblemType::WordProblem);
        let equation = dataset.get_problems_by_type(&MathProblemType::Equation);
        
        assert!(arithmetic.len() > 0);
        assert!(algebra.len() > 0);
        assert!(word.len() > 0);
        assert!(equation.len() > 0);
    }

    #[test]
    fn test_difficulty_filtering() {
        let dataset = MathDataset::create_default_dataset();
        
        let elementary = dataset.get_problems_by_difficulty(&DifficultyLevel::Elementary);
        let intermediate = dataset.get_problems_by_difficulty(&DifficultyLevel::Intermediate);
        
        assert!(elementary.len() > 0);
        assert!(intermediate.len() > 0);
    }
}