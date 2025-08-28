//! # Text Generation Demo
//!
//! Demonstrates the text generation capabilities of the DeepSeek R1 implementation.

use ds_r1_rs::{
    inference::{
        engine::{InferenceEngine, ProblemType},
        generation::GenerationConfig,
        sampling::SamplingConfig,
    },
    model::{config::ModelConfig, DeepSeekR1Model},
    utils::tokenizer::TokenizerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DeepSeek R1 Text Generation Demo");
    println!("=================================\n");

    // Create model and inference engine
    let model_config = ModelConfig::default();
    let model = DeepSeekR1Model::new(model_config)?;
    
    let tokenizer_config = TokenizerConfig::default();
    let sampling_config = SamplingConfig::default();
    let mut generation_config = GenerationConfig::default();
    generation_config.max_tokens = 100;
    generation_config.temperature = 0.8;

    let mut engine = InferenceEngine::with_configs(
        model,
        tokenizer_config,
        sampling_config,
        generation_config,
    )?;

    // Test basic text generation
    println!("1. Basic Text Generation");
    println!("------------------------");
    let prompt = "The future of artificial intelligence is";
    println!("Prompt: {}", prompt);
    
    match engine.generate_text(prompt) {
        Ok(text) => println!("Generated: {}\n", text),
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    // Test reasoning-aware generation
    println!("2. Reasoning-Aware Generation");
    println!("-----------------------------");
    let reasoning_prompt = "Explain the process of photosynthesis";
    println!("Prompt: {}", reasoning_prompt);
    
    match engine.generate_with_reasoning(reasoning_prompt) {
        Ok(reasoning_output) => {
            println!("Reasoning steps: {}", reasoning_output.reasoning_steps);
            for (i, step) in reasoning_output.thinking_chain.iter().enumerate() {
                println!("  {}. {}", i + 1, step);
            }
            println!("Final answer: {}\n", reasoning_output.final_answer);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    // Test problem-solving interface
    println!("3. Problem-Solving Interface");
    println!("----------------------------");
    
    let math_problem = "If a train travels 60 mph for 2.5 hours, how far does it go?";
    println!("Math Problem: {}", math_problem);
    
    match engine.solve_math_problem(math_problem) {
        Ok(solution) => {
            println!("Solution steps: {}", solution.reasoning_steps);
            println!("Final answer: {}\n", solution.final_answer);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    let code_snippet = r#"
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"#;
    println!("Code to explain:");
    println!("{}", code_snippet);
    
    match engine.explain_code(code_snippet) {
        Ok(explanation) => {
            println!("Explanation steps: {}", explanation.reasoning_steps);
            println!("Final explanation: {}\n", explanation.final_answer);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)\n", e),
    }

    // Test different problem types
    println!("4. Different Problem Types");
    println!("--------------------------");
    
    let problems = [
        ("Mathematical", "What is 15% of 240?", ProblemType::Mathematical),
        ("Logical", "All birds can fly. Penguins are birds. Can penguins fly?", ProblemType::Logical),
        ("General", "How do you make a good first impression?", ProblemType::General),
    ];

    for (problem_type, problem, ptype) in &problems {
        println!("{} Problem: {}", problem_type, problem);
        match engine.solve_problem(problem, ptype.clone()) {
            Ok(solution) => {
                println!("Solution: {}", solution.final_answer);
            }
            Err(e) => println!("Error: {} (Expected - model not fully implemented)", e),
        }
        println!();
    }

    // Test generation configuration
    println!("5. Generation Configuration");
    println!("---------------------------");
    
    let mut custom_config = GenerationConfig::default();
    custom_config.max_tokens = 50;
    custom_config.temperature = 0.2; // More deterministic
    
    println!("Using custom config (max_tokens: {}, temperature: {})", 
             custom_config.max_tokens, custom_config.temperature);
    
    match engine.generate_text_with_config("Write a haiku about programming", &custom_config) {
        Ok(output) => {
            println!("Generated text: {}", output.text);
            println!("Tokens generated: {}", output.tokens_generated);
            println!("Generation time: {}ms", output.generation_time_ms);
        }
        Err(e) => println!("Error: {} (Expected - model not fully implemented)", e),
    }

    println!("\nDemo completed!");
    println!("Note: This demo shows the text generation interface structure.");
    println!("Actual generation will work once the transformer model is fully implemented.");

    Ok(())
}