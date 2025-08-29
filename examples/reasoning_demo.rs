//! # Reasoning Demo
//!
//! Demonstrates the reasoning capabilities of the DeepSeek R1 implementation.

use ds_r1_rs::{
    inference::engine::{InferenceEngine, ProblemType},
    model::{config::ModelConfig, transformer::DeepSeekR1Model},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DeepSeek R1 Reasoning Demo");
    println!("==========================\n");

    // Initialize the model and inference engine
    let config = ModelConfig::default();
    let model = DeepSeekR1Model::new(config)?;
    let mut engine = InferenceEngine::new(model)?;

    // Note: Since the actual model implementation is not complete,
    // these examples will demonstrate the interface but may not produce
    // meaningful results until the model forward pass is implemented.

    println!("1. Mathematical Problem Solving");
    println!("-------------------------------");
    let math_problem = "What is the sum of the first 5 positive integers?";
    println!("Problem: {}", math_problem);

    match engine.solve_math_problem(math_problem) {
        Ok(result) => {
            println!("Reasoning steps: {}", result.reasoning_steps);
            println!("Final answer: {}", result.final_answer);
            println!("Confidence: {:.2}\n", result.confidence);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    println!("2. Code Explanation");
    println!("-------------------");
    let code_sample = r#"
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
"#;
    println!("Code to explain:\n{}", code_sample);

    match engine.explain_code(code_sample) {
        Ok(result) => {
            println!("Reasoning steps: {}", result.reasoning_steps);
            println!("Explanation: {}", result.final_answer);
            println!("Confidence: {:.2}\n", result.confidence);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    println!("3. Logical Reasoning");
    println!("--------------------");
    let logic_problem =
        "If all cats are mammals, and Fluffy is a cat, what can we conclude about Fluffy?";
    println!("Problem: {}", logic_problem);

    match engine.solve_logical_problem(logic_problem) {
        Ok(result) => {
            println!("Reasoning steps: {}", result.reasoning_steps);
            println!("Conclusion: {}", result.final_answer);
            println!("Confidence: {:.2}\n", result.confidence);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    println!("4. General Problem Solving");
    println!("--------------------------");
    let general_problem = "How would you organize a small library of 1000 books?";
    println!("Problem: {}", general_problem);

    match engine.solve_problem(general_problem, ProblemType::General) {
        Ok(result) => {
            println!("Reasoning steps: {}", result.reasoning_steps);
            println!("Solution: {}", result.final_answer);
            println!("Confidence: {:.2}\n", result.confidence);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    println!("5. Reasoning-Aware Text Generation");
    println!("-----------------------------------");
    let reasoning_prompt = "Explain why the sky appears blue during the day.";
    println!("Prompt: {}", reasoning_prompt);

    match engine.generate_with_reasoning(reasoning_prompt) {
        Ok(result) => {
            println!("Thinking process:");
            for (i, step) in result.thinking_chain.iter().enumerate() {
                println!("  {}. {}", i + 1, step);
            }
            println!("Final response: {}", result.final_answer);
            println!("Confidence: {:.2}\n", result.confidence);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    println!("Demo completed!");
    println!("\nNote: This demo shows the reasoning interface structure.");
    println!("Actual reasoning will work once the transformer model is fully implemented.");

    Ok(())
}
