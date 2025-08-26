use ds_r1_rs::{DeepSeekR1Model, InferenceEngine, ModelConfig};
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
        println!();
        println!("Examples:");
        println!("  cargo run -- config");
        println!("  cargo run -- test");
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
        _ => {
            println!("Unknown command: {}", args[1]);
            println!("Use 'config', 'version', or 'test'");
        }
    }

    Ok(())
}
