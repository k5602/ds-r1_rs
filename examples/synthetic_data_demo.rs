//! # Synthetic Data Generation Demo
//!
//! Demonstrates the synthetic training data generation capabilities.

use ds_r1_rs::training::{DataLoader, SyntheticDataGenerator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DeepSeek R1 Synthetic Data Generation Demo ===\n");

    let mut generator = SyntheticDataGenerator::new();

    // Generate different types of problems
    println!("🧮 Mathematical Reasoning Problems:");
    let math_problems = generator.generate_math_problems(3);
    for (i, problem) in math_problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem.input);
        if let Some(reasoning) = &problem.reasoning_chain {
            println!("Reasoning:");
            for step in reasoning {
                println!("  - {}", step);
            }
        }
        println!("Answer: {}\n", problem.target);
    }

    println!("💻 Code Understanding Examples:");
    let code_examples = generator.generate_code_examples(2);
    for (i, example) in code_examples.iter().enumerate() {
        println!("Example {}: {}", i + 1, example.input);
        if let Some(reasoning) = &example.reasoning_chain {
            println!("Reasoning:");
            for step in reasoning {
                println!("  - {}", step);
            }
        }
        println!("Explanation: {}\n", example.target);
    }

    println!("🧠 Logical Reasoning Tasks:");
    let logic_problems = generator.generate_logic_problems(2);
    for (i, problem) in logic_problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem.input);
        if let Some(reasoning) = &problem.reasoning_chain {
            println!("Reasoning:");
            for step in reasoning {
                println!("  - {}", step);
            }
        }
        println!("Answer: {}\n", problem.target);
    }

    // Demonstrate data loading
    println!("📦 Data Loading Demo:");
    let mixed_dataset = generator.generate_mixed_dataset(10);
    println!("Generated {} mixed examples", mixed_dataset.len());

    let mut loader = DataLoader::new(mixed_dataset, 3, true);
    println!(
        "Batch size: 3, Total batches: {}",
        loader.batches_per_epoch()
    );

    let mut batch_count = 0;
    while let Some(batch) = loader.next_batch() {
        batch_count += 1;
        println!("Batch {}: {} examples", batch_count, batch.batch_size);

        let by_type = batch.split_by_type();
        for (problem_type, examples) in by_type {
            println!("  {:?}: {} examples", problem_type, examples.len());
        }
    }

    println!("\n✅ Synthetic data generation demo completed!");
    Ok(())
}
