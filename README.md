# 🧠 DeepSeek R1 Rust Implementation

A prototype implementation of DeepSeek R1-inspired reasoning model in Rust for educational and research purposes.


## 🏗️ Architecture

```
ds-r1_rs/
├── src/
│   ├── lib.rs              # Main library entry point
│   ├── main.rs             # CLI interface
│   ├── model/              # Core model components
│   │   ├── config.rs       # Model configuration
│   │   ├── transformer.rs  # Main model structure
│   │   ├── attention.rs    # MLA and standard attention
│   │   ├── moe.rs          # Mixture of Experts
│   │   ├── embeddings.rs   # Token and positional embeddings
│   │   └── layers.rs       # Neural network layers
│   ├── training/           # Training infrastructure
│   │   ├── trainer.rs      # Training implementations
│   │   ├── data.rs         # Training data structures
│   │   ├── optimizer.rs    # Optimization algorithms
│   │   └── loss.rs         # Loss functions
│   ├── inference/          # Inference capabilities
│   │   ├── engine.rs       # Main inference engine
│   │   ├── reasoning.rs    # Reasoning chain processing
│   │   ├── sampling.rs     # Text sampling strategies
│   │   └── generation.rs   # Text generation utilities
│   └── utils/              # Common utilities
│       ├── error.rs        # Error handling
│       ├── math.rs         # Mathematical functions
│       ├── tokenizer.rs    # Text tokenization
│       └── evaluation.rs   # Model evaluation
├── examples/
│   └── config_demo.rs      # Configuration demonstration
└── tests/                  # Integration tests
```

## 🛠️ Usage

### CLI Commands

```bash
# Show help
cargo run

# Display model configuration
cargo run -- config

# Show version information
cargo run -- version

# Run basic functionality tests
cargo run -- test
```

### Development Commands

```bash
# Run all unit tests
cargo test

# Check code compilation
cargo check

# Build optimized binary
cargo build --release

# Run configuration demo
cargo run --example config_demo
```

## 📋 Model Configuration

The model supports the following key parameters:

- **Architecture**: 8 layers, 8 heads, 512 hidden size
- **MLA**: Multi-head Latent Attention with 0.5 compression ratio
- **MoE**: 8 experts with 2 experts per token
- **Reasoning**: Special thinking tokens for reasoning chains
- **Sequence Length**: Up to 2048 tokens

## 🧪 Testing

All components include comprehensive unit tests:

```bash
cargo test                    # Run all tests
cargo test test_config       # Run specific test
cargo test --verbose         # Verbose output
```

## 🔧 Dependencies

- **serde**: Configuration serialization
- **thiserror**: Error handling
- **rand**: Random number generation
- **log/env_logger**: Logging infrastructure

## 📝 Next Steps

The foundation is complete. Next tasks will implement:

1. **Core Tensor Operations**: linear algebra and tensor operations
2. **Attention Mechanisms**: Multi-head Latent Attention (MLA) implementation
3. **Transformer Layers**: Complete transformer block with residual connections
4. **Mixture of Experts**: MoE routing and expert networks
5. **Training Infrastructure**: Supervised and reinforcement learning
6. **Inference Engine**: Text generation and reasoning capabilities

## 🤝 Contributing

This is an educational prototype. The codebase follows Rust best practices with:

- Comprehensive error handling
- Extensive documentation
- Unit tests for all components
- Modular architecture
- Type safety and memory safety

## 📄 License

MIT License - Educational and research use.

Made with Insistence by Khaled