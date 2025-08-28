use ds_r1_rs::inference::engine::InferenceEngine;
use ds_r1_rs::inference::reasoning::{
    ReasoningAnalysis, ReasoningSection, StructuredReasoningOutput,
};
use ds_r1_rs::utils::evaluation::EvaluationHarness;
use ds_r1_rs::{DeepSeekR1Model, ModelConfig};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("🧠 DeepSeek R1 Rust Implementation");
        println!("A prototype implementation of DeepSeek R1-inspired reasoning model in Rust");
        println!();
        println!("Usage: {} <command>", args[0]);
        println!();
        println!("Commands:");
        println!("  config    - Show default model configuration");
        println!("  version   - Show version and build information");
        println!("  test      - Run basic functionality tests");
        println!("  generate  - Generate text from a prompt");
        println!("  eval      - Run reasoning benchmarks");
        println!();
        println!("Examples:");
        println!("  cargo run -- config");
        println!("  cargo run -- test");
        println!("  cargo run -- generate \"Explain Rust ownership\"");
        println!("  cargo run -- eval");
        println!("  cargo run --example config_demo");
        println!();
        println!("For development:");
        println!("  cargo test           - Run all unit tests");
        println!("  cargo check          - Check code compilation");
        println!("  cargo build --release - Build optimized binary");
        println!("Made with insistence by khaled");
        return Ok(());
    }

    match args[1].as_str() {
        "config" => {
            let config = ModelConfig::default();
            println!("🔧 Default Model Configuration:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Architecture:");
            println!("  • Vocabulary size: {}", config.vocab_size);
            println!("  • Hidden size: {}", config.hidden_size);
            println!("  • Number of layers: {}", config.num_layers);
            println!("  • Number of heads: {}", config.num_heads);
            println!("  • Intermediate size: {}", config.intermediate_size);
            println!("  • Max sequence length: {}", config.max_seq_len);
            println!();
            println!("Multi-head Latent Attention (MLA):");
            println!("  • KV compression ratio: {}", config.kv_compression_ratio);
            println!("  • RoPE theta: {}", config.rope_theta);
            println!("  • Head dimension: {}", config.head_dim());
            println!(
                "  • Compressed KV dimension: {}",
                config.compressed_kv_dim()
            );
            println!();
            println!("Mixture of Experts (MoE):");
            println!("  • Number of experts: {}", config.num_experts);
            println!("  • Experts per token: {}", config.experts_per_token);
            println!();
            println!("Reasoning:");
            println!("  • Thinking token ID: {}", config.thinking_token_id);
            println!("  • Max reasoning steps: {}", config.max_reasoning_steps);
            println!();
            println!("Training:");
            println!("  • Dropout probability: {}", config.dropout_prob);
            println!("  • Layer norm epsilon: {}", config.layer_norm_eps);
        }
        "version" => {
            println!("🚀 DeepSeek R1 Rust Implementation");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Version: {}", ds_r1_rs::VERSION);
            println!("Edition: Rust 2021");
            println!("Purpose: Educational prototype and research");
            println!();
            println!("Features implemented:");
            println!("  ✓ Project structure and configuration system");
            println!("  ✓ Module architecture (model, training, inference, utils)");
            println!("  ✓ Error handling and validation");
            println!("  ✓ Basic CLI interface");
            println!("  ✓ Unit tests and examples");
            println!();
            println!("Next steps: Implement transformer layers, attention mechanisms, and MoE");
        }
        "test" => {
            println!("🧪 Running Basic Functionality Tests");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // Test configuration
            print!("Testing configuration system... ");
            let config = ModelConfig::default();
            match config.validate() {
                Ok(()) => println!("✓ PASSED"),
                Err(e) => {
                    println!("✗ FAILED: {}", e);
                    return Err(e.into());
                }
            }

            // Test model creation
            print!("Testing model creation... ");
            match DeepSeekR1Model::new(config.clone()) {
                Ok(_model) => println!("✓ PASSED"),
                Err(e) => {
                    println!("✗ FAILED: {}", e);
                    return Err(e.into());
                }
            }

            // Test configuration serialization
            print!("Testing config serialization... ");
            match serde_json::to_string(&config) {
                Ok(_json) => println!("✓ PASSED"),
                Err(e) => {
                    println!("✗ FAILED: {}", e);
                    return Err(e.into());
                }
            }

            println!();
            println!("🎉 All basic functionality tests passed!");
            println!("The project foundation is ready for implementing model components.");
        }
        "generate" => {
            if args.len() < 3 {
                println!("Usage: {} generate <prompt>", args[0]);
                return Ok(());
            }
            let prompt = args[2..].join(" ");
            let config = ModelConfig::default();
            let model = DeepSeekR1Model::new(config)?;
            let mut engine = InferenceEngine::new(model)?;
            match engine.generate_text(&prompt) {
                Ok(text) => {
                    println!("Generated:");
                    println!("{}", text);
                }
                Err(e) => {
                    println!("Generation failed: {}", e);
                    return Err(e.into());
                }
            }
        }
        "eval" => {
            println!("🧪 Running Reasoning Benchmarks");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            let config = ModelConfig::default();
            let model = DeepSeekR1Model::new(config.clone())?;
            let mut engine = InferenceEngine::new(model)?;
            let harness = EvaluationHarness::new();

            for b in harness.get_benchmarks() {
                let mut infer =
                    |problem: &str| -> ds_r1_rs::utils::error::Result<StructuredReasoningOutput> {
                        let reasoning = engine.generate_with_reasoning(problem)?;
                        let analysis = ReasoningAnalysis::new(&reasoning.thinking_chain);
                        let parsed_sections = reasoning
                            .thinking_chain
                            .iter()
                            .map(|s| ReasoningSection::new(s.clone()))
                            .collect();
                        Ok(StructuredReasoningOutput {
                            reasoning_output: reasoning,
                            analysis,
                            parsed_sections,
                        })
                    };

                match harness.evaluate_comprehensive(&b.name, &mut infer) {
                    Ok(results) => {
                        println!("Benchmark: {}", results.benchmark_name);
                        println!("  Total problems: {}", results.total_problems);
                        println!("  Solved correctly: {}", results.solved_correctly);
                        println!(
                            "  Avg overall quality: {:.2}",
                            results.average_metrics.overall_quality
                        );
                        println!(
                            "  Tokens/sec: {:.2}",
                            results.performance_metrics.tokens_per_second
                        );
                        println!();
                    }
                    Err(e) => {
                        println!("Evaluation failed for {}: {}", b.name, e);
                    }
                }
            }
        }
        _ => {
            println!("Unknown command: {}", args[1]);
            println!("Use 'config', 'version', 'test', 'generate', or 'eval'");
        }
    }

    Ok(())
}
