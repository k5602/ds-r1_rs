//! # Training Infrastructure Demo
//!
//! Demonstrates both supervised and reinforcement learning training capabilities.

use ds_r1_rs::model::{DeepSeekR1Model, ModelConfig};
use ds_r1_rs::training::{
    BasicTrainer, DataLoader, OptimizerConfig, RLTrainer, SyntheticDataGenerator, TrainingBatch,
    TrainingExample, ProblemType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DeepSeek R1 Training Infrastructure Demo ===\n");

    // Initialize model
    let config = ModelConfig::default();
    let model = DeepSeekR1Model::new(config.clone())?;
    println!("✅ Model initialized with config: {:?}\n", model.config());

    // Generate synthetic training data
    let mut data_generator = SyntheticDataGenerator::new();
    let training_examples = data_generator.generate_mixed_dataset(20);
    println!("📊 Generated {} training examples", training_examples.len());

    // Split into train/test
    let split_point = training_examples.len() * 8 / 10;
    let train_examples = training_examples[..split_point].to_vec();
    let test_examples = training_examples[split_point..].to_vec();
    println!("   Training: {} examples", train_examples.len());
    println!("   Testing: {} examples\n", test_examples.len());

    // Demonstrate supervised learning
    println!("🎓 Supervised Learning Demo:");
    let supervised_model = DeepSeekR1Model::new(config.clone())?;
    demonstrate_supervised_learning(supervised_model, &train_examples, &test_examples)?;

    // Demonstrate reinforcement learning
    println!("\n🧠 Reinforcement Learning Demo:");
    let rl_model = DeepSeekR1Model::new(config)?;
    demonstrate_reinforcement_learning(rl_model, &train_examples, &test_examples)?;

    println!("\n✅ Training infrastructure demo completed!");
    Ok(())
}

fn demonstrate_supervised_learning(
    model: DeepSeekR1Model,
    train_examples: &[TrainingExample],
    test_examples: &[TrainingExample],
) -> Result<(), Box<dyn std::error::Error>> {
    // Create trainer with custom optimizer config
    let optimizer_config = OptimizerConfig {
        learning_rate: 0.001,
        weight_decay: 0.01,
        ..OptimizerConfig::default()
    };
    
    let mut trainer = BasicTrainer::with_optimizer_config(model, optimizer_config)?;
    println!("   Created supervised trainer");

    // Create data loader
    let mut data_loader = DataLoader::new(train_examples.to_vec(), 4, true);
    println!("   Created data loader with batch size 4");

    // Training loop
    println!("   Training for 3 epochs...");
    for epoch in 1..=3 {
        data_loader.reset();
        let mut epoch_loss = 0.0;
        let mut epoch_accuracy = 0.0;
        let mut batch_count = 0;

        while let Some(batch) = data_loader.next_batch() {
            let metrics = trainer.train_step(&batch)?;
            epoch_loss += metrics.loss;
            epoch_accuracy += metrics.accuracy;
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count as f32;
        let avg_accuracy = epoch_accuracy / batch_count as f32;
        
        println!("     Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", 
                epoch, avg_loss, avg_accuracy * 100.0);
    }

    // Evaluation
    println!("   Evaluating on test set...");
    let test_metrics = trainer.evaluate(test_examples)?;
    println!("     Test Loss: {:.4}", test_metrics.loss);
    println!("     Test Accuracy: {:.2}%", test_metrics.accuracy * 100.0);
    println!("     Test Perplexity: {:.4}", test_metrics.perplexity);

    Ok(())
}

fn demonstrate_reinforcement_learning(
    model: DeepSeekR1Model,
    train_examples: &[TrainingExample],
    test_examples: &[TrainingExample],
) -> Result<(), Box<dyn std::error::Error>> {
    // Create RL trainer with lower learning rate
    let optimizer_config = OptimizerConfig {
        learning_rate: 1e-5,
        weight_decay: 0.001,
        ..OptimizerConfig::default()
    };
    
    let mut rl_trainer = RLTrainer::with_optimizer_config(model, optimizer_config)?;
    println!("   Created RL trainer with REINFORCE algorithm");

    // Create data loader for RL training
    let mut data_loader = DataLoader::new(train_examples.to_vec(), 2, true);
    println!("   Created data loader with batch size 2");

    // RL training loop
    println!("   Training with REINFORCE for 3 episodes...");
    for episode in 1..=3 {
        data_loader.reset();
        let mut episode_reward = 0.0;
        let mut episode_accuracy = 0.0;
        let mut batch_count = 0;

        while let Some(batch) = data_loader.next_batch() {
            let metrics = rl_trainer.train_step(&batch)?;
            episode_reward += metrics.loss; // In RL, we track negative loss as reward
            episode_accuracy += metrics.accuracy;
            batch_count += 1;
        }

        let avg_reward = -episode_reward / batch_count as f32; // Convert back to positive reward
        let avg_accuracy = episode_accuracy / batch_count as f32;
        let baseline = rl_trainer.baseline();
        
        println!("     Episode {}: Avg Reward = {:.4}, Accuracy = {:.2}%, Baseline = {:.4}", 
                episode, avg_reward, avg_accuracy * 100.0, baseline);
    }

    // RL evaluation
    println!("   Evaluating RL policy on test set...");
    let rl_metrics = rl_trainer.evaluate(test_examples)?;
    println!("     Average Reward: {:.4}", rl_metrics.average_reward);
    println!("     Accuracy: {:.2}%", rl_metrics.accuracy * 100.0);
    println!("     Reasoning Quality: {:.4}", rl_metrics.reasoning_quality);
    println!("     Current Baseline: {:.4}", rl_metrics.baseline);

    // Show detailed RL metrics
    println!("   Detailed RL Evaluation:");
    rl_metrics.display();

    Ok(())
}

/// Demonstrate different problem types
#[allow(dead_code)]
fn demonstrate_problem_types() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Problem Type Analysis:");
    
    let mut generator = SyntheticDataGenerator::new();
    let examples = generator.generate_mixed_dataset(12);
    
    let batch = TrainingBatch::new(examples);
    let by_type = batch.split_by_type();
    
    for (problem_type, examples) in by_type {
        println!("   {:?}: {} examples", problem_type, examples.len());
        
        // Show one example of each type
        if let Some(example) = examples.first() {
            println!("     Example: {}", example.input);
            println!("     Target: {}", example.target);
            if let Some(reasoning) = &example.reasoning_chain {
                println!("     Reasoning steps: {}", reasoning.len());
            }
        }
        println!();
    }
    
    Ok(())
}