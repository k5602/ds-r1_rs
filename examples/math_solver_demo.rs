//! # Mathematical Problem Solver Demo
//!
//! Demonstrates structured mathematical reasoning and problem solving capabilities.

use ds_r1_rs::{
    inference::{
        engine::InferenceEngine,
        math_solver::{MathProblemSolver, MathProblemType},
    },
    model::{config::ModelConfig, transformer::DeepSeekR1Model},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DeepSeek R1 Mathematical Problem Solver Demo");
    println!("============================================\n");

    // Initialize the model and inference engine
    let config = ModelConfig::default();
    let model = DeepSeekR1Model::new(config.clone())?;
    let mut engine = InferenceEngine::new(model)?;

    // Initialize the mathematical problem solver
    let mut math_solver =
        MathProblemSolver::new(config.thinking_token_id, config.thinking_token_id + 1);

    println!("🧮 Mathematical Problem Solving Demonstrations");
    println!("===============================================\n");

    // Demonstrate different types of mathematical problems
    demonstrate_arithmetic_problems(&mut math_solver)?;
    demonstrate_algebraic_problems(&mut math_solver)?;
    demonstrate_word_problems(&mut math_solver)?;
    demonstrate_equation_solving(&mut math_solver)?;
    demonstrate_inference_engine_integration(&mut engine)?;

    println!("✅ Mathematical reasoning demo completed!");
    println!("\nNote: This demo shows the mathematical reasoning structure.");
    println!(
        "Full reasoning capabilities will be available once the transformer model is complete."
    );

    Ok(())
}

/// Demonstrate arithmetic problem solving
fn demonstrate_arithmetic_problems(
    solver: &mut MathProblemSolver,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Arithmetic Problems");
    println!("----------------------");

    let arithmetic_problems = vec![
        "What is 15 + 27?",
        "Calculate 84 - 39",
        "What is 12 times 8?",
        "Divide 144 by 12",
    ];

    for (i, problem) in arithmetic_problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem);

        match solver.solve_problem(problem) {
            Ok(solution) => {
                println!("Type: {:?}", solution.problem_type);
                println!("Steps:");
                for step in &solution.reasoning_steps {
                    println!("  {}. {}", step.step_number, step.description);
                    if let Some(calc) = &step.calculation {
                        println!("     Calculation: {}", calc);
                    }
                    if let Some(result) = &step.result {
                        println!("     Result: {}", result);
                    }
                }
                if let Some(answer) = &solution.final_answer {
                    println!("Final Answer: {}", answer);
                }
                println!("Confidence: {:.1}%", solution.confidence * 100.0);
            }
            Err(e) => println!("Error solving problem: {}", e),
        }
        println!();
    }

    Ok(())
}

/// Demonstrate algebraic problem solving
fn demonstrate_algebraic_problems(
    solver: &mut MathProblemSolver,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Algebraic Problems");
    println!("---------------------");

    let algebra_problems = vec![
        "Find the value of x when 2x + 5 = 13",
        "Solve for x: x + 3 = 7",
        "What is x if 3x = 15?",
        "Find x when x - 4 = 10",
    ];

    for (i, problem) in algebra_problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem);

        match solver.solve_problem(problem) {
            Ok(solution) => {
                println!("Type: {:?}", solution.problem_type);
                println!("Solution Process:");
                for step in &solution.reasoning_steps {
                    println!("  {}. {}", step.step_number, step.description);
                    if let Some(calc) = &step.calculation {
                        println!("     Work: {}", calc);
                    }
                    if let Some(result) = &step.result {
                        println!("     Result: {}", result);
                    }
                }
                if let Some(answer) = &solution.final_answer {
                    println!("Final Answer: {}", answer);
                }
                if let Some(verification) = &solution.verification {
                    println!("Verification: {}", verification);
                }
                println!("Confidence: {:.1}%", solution.confidence * 100.0);
            }
            Err(e) => println!("Error solving problem: {}", e),
        }
        println!();
    }

    Ok(())
}

/// Demonstrate word problem solving
fn demonstrate_word_problems(
    solver: &mut MathProblemSolver,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Word Problems");
    println!("----------------");

    let word_problems = vec![
        "John has 15 apples and Mary has 23 apples. How many apples do they have in total?",
        "A store sold 45 items in the morning and 38 items in the afternoon. What is the total number of items sold?",
        "Sarah bought 8 books and each book costs 12 dollars. How much did she spend in total?",
        "There are 100 students in a school. If 35 are boys, how many are girls?",
    ];

    for (i, problem) in word_problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem);

        match solver.solve_problem(problem) {
            Ok(solution) => {
                println!("Type: {:?}", solution.problem_type);
                println!("Solution Steps:");
                for step in &solution.reasoning_steps {
                    println!("  {}. {}", step.step_number, step.description);
                    if let Some(calc) = &step.calculation {
                        println!("     Expression: {}", calc);
                    }
                    if let Some(result) = &step.result {
                        println!("     Answer: {}", result);
                    }
                }
                if let Some(answer) = &solution.final_answer {
                    println!("Final Answer: {}", answer);
                }
                println!("Confidence: {:.1}%", solution.confidence * 100.0);
            }
            Err(e) => println!("Error solving problem: {}", e),
        }
        println!();
    }

    Ok(())
}

/// Demonstrate equation solving
fn demonstrate_equation_solving(
    solver: &mut MathProblemSolver,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Equation Solving");
    println!("-------------------");

    let equation_problems = vec![
        "Solve: 2x + 5 = 13",
        "Find x: x + 3 = 7",
        "Solve: 3x = 15",
        "Find the value: 4x - 8 = 12",
    ];

    for (i, problem) in equation_problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem);

        match solver.solve_problem(problem) {
            Ok(solution) => {
                println!("Type: {:?}", solution.problem_type);
                println!("Solution Method:");
                for step in &solution.reasoning_steps {
                    println!("  {}. {}", step.step_number, step.description);
                    if let Some(result) = &step.result {
                        println!("     Result: {}", result);
                    }
                }
                if let Some(answer) = &solution.final_answer {
                    println!("Final Answer: {}", answer);
                }
                if let Some(verification) = &solution.verification {
                    println!("Verification: {}", verification);
                }
                println!("Confidence: {:.1}%", solution.confidence * 100.0);
            }
            Err(e) => println!("Error solving problem: {}", e),
        }
        println!();
    }

    Ok(())
}

/// Demonstrate integration with inference engine
fn demonstrate_inference_engine_integration(
    engine: &mut InferenceEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Inference Engine Integration");
    println!("-------------------------------");

    let math_problems = vec![
        "What is 25% of 80?",
        "If a rectangle has length 12 and width 8, what is its area?",
        "Calculate the average of 15, 20, 25, and 30",
        "What is the perimeter of a square with side length 7?",
    ];

    for (i, problem) in math_problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem);

        // Use the inference engine's math problem solving capability
        match engine.solve_math_problem_detailed(problem) {
            Ok(solution) => {
                println!("Problem Type: Mathematical");
                println!("Reasoning Process:");
                for (j, step) in solution.reasoning_steps.iter().enumerate() {
                    println!("  {}. {}", j + 1, step);
                }
                if let Some(answer) = &solution.final_answer {
                    println!("Final Answer: {}", answer);
                }
                println!("Confidence: {:.1}%", solution.confidence * 100.0);
            }
            Err(e) => println!("Error: {} (Expected - model not fully implemented)", e),
        }
        println!();
    }

    Ok(())
}

/// Demonstrate mathematical formatting and answer extraction
fn demonstrate_mathematical_formatting() {
    println!("6. Mathematical Formatting Examples");
    println!("-----------------------------------");

    // Show examples of proper mathematical notation
    let examples = vec![
        ("Basic Arithmetic", "15 + 27 = 42"),
        ("Algebraic Solution", "2x + 5 = 13 → 2x = 8 → x = 4"),
        ("Fraction", "3/4 = 0.75"),
        ("Percentage", "25% of 80 = 0.25 × 80 = 20"),
        (
            "Area Formula",
            "Area = length × width = 12 × 8 = 96 square units",
        ),
    ];

    for (category, example) in examples {
        println!("{}: {}", category, example);
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_solver_demo_components() {
        // Test that the demo components can be created
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config.clone()).unwrap();
        let _engine = InferenceEngine::new(model).unwrap();

        let _math_solver =
            MathProblemSolver::new(config.thinking_token_id, config.thinking_token_id + 1);
    }

    #[test]
    fn test_problem_type_detection() {
        let config = ModelConfig::default();
        let solver = MathProblemSolver::new(config.thinking_token_id, config.thinking_token_id + 1);

        assert_eq!(
            solver.detect_problem_type("What is 2 + 2?"),
            MathProblemType::Arithmetic
        );
        assert_eq!(
            solver.detect_problem_type("Solve for x"),
            MathProblemType::Algebra
        );
        assert_eq!(
            solver.detect_problem_type("John has 5 apples"),
            MathProblemType::WordProblem
        );
        assert_eq!(
            solver.detect_problem_type("2x + 5 = 13"),
            MathProblemType::Equation
        );
    }
}
