use ds_r1_rs::ModelConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DeepSeek R1 Configuration Demo");
    
    // Create default configuration
    let config = ModelConfig::default();
    println!("Default configuration created:");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Number of layers: {}", config.num_layers);
    println!("  Number of heads: {}", config.num_heads);
    println!("  Max sequence length: {}", config.max_seq_len);
    println!("  Number of experts: {}", config.num_experts);
    println!("  Experts per token: {}", config.experts_per_token);
    
    // Validate configuration
    match config.validate() {
        Ok(()) => println!("✓ Configuration is valid"),
        Err(e) => println!("✗ Configuration validation failed: {}", e),
    }
    
    // Test derived values
    println!("Derived values:");
    println!("  Head dimension: {}", config.head_dim());
    println!("  Compressed KV dimension: {}", config.compressed_kv_dim());
    
    // Save configuration to file
    let config_path = "test_config.json";
    match config.save_to_file(config_path) {
        Ok(()) => {
            println!("✓ Configuration saved to {}", config_path);
            
            // Load configuration from file
            match ModelConfig::load_from_file(config_path) {
                Ok(loaded_config) => {
                    println!("✓ Configuration loaded from file");
                    println!("  Loaded vocab size: {}", loaded_config.vocab_size);
                    
                    // Clean up
                    std::fs::remove_file(config_path).ok();
                }
                Err(e) => println!("✗ Failed to load configuration: {}", e),
            }
        }
        Err(e) => println!("✗ Failed to save configuration: {}", e),
    }
    
    Ok(())
}