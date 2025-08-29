# 🧠 DeepSeek R1 (Rust) — Research-Grade Reasoning Model Prototype

A Rust implementation of a DeepSeek R1–inspired reasoning model focused on clarity, testability, and strong engineering practices. This project is designed to be an impressive portfolio piece: it includes a modular transformer architecture, reasoning-aware inference, evaluation harness, examples, comprehensive tests, and CI.

Highlights:

- Fully-typed Rust 2024 crate with modules for model, inference, training, and utilities
- Transformer stack with rotary embeddings, standard attention, pre-norm layers, and an LM head
- MLA (Multi-head Latent Attention) and MoE (Mixture of Experts) components implemented and tested
- Reasoning-aware generation pipeline with <think>…</think> parsing and structured analysis
- Evaluation harness for benchmarks across math, logic, programming, and general reasoning
- Examples that compile and run via cargo
- GitHub Actions CI with build, lint, test, examples, and coverage

## 🚀 Quick Start

Prerequisites: Rust (stable), Cargo.

Build:

```bash
cargo build
```

Run CLI:

```bash
# Help (shows available commands)
cargo run
```

Core commands:

```bash
# Show default model configuration
cargo run -- config

# Show version and build info
cargo run -- version

# Run basic checks and smoke tests
cargo run -- test

# Generate text from a prompt (uses simple model forward)
cargo run -- generate "Explain Rust ownership in simple terms"

# Evaluate reasoning benchmarks (math, logic, programming, general)
cargo run -- eval

# Export evaluation results as JSON (for dashboards)
cargo run -- eval --json > results.json

# Save current model weights (full)
cargo run -- save-weights ckpt.json

# Save only lm_head parameters; exclude embeddings
cargo run -- save-weights ckpt.json --include lm_head --exclude embeddings

# Save a small demo-size checkpoint (size-conscious)
cargo run -- save-weights ckpt-small.json --demo-small

# Load weights and generate deterministically (temperature=0)
cargo run -- load-weights ckpt.json "Explain Rust ownership"

# Load only lm_head from checkpoint, allowing missing others
cargo run -- load-weights ckpt.json --allow-missing --include lm_head "Explain Rust ownership"
```

Run examples:

```bash
cargo run --example config_demo
cargo run --example generation_demo
cargo run --example math_solver_demo
cargo run --example training_demo
```

Tests + checks:

```bash
# Unit + doc tests
cargo test

# Integration tests (CLI)
cargo test --test cli_integration
# Optional heavier integration tests
cargo test --test cli_integration -- --ignored

# Lints/format
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check

# Benchmarks (Criterion)
cargo bench --bench decoding -- --warm-up-time 0.5 --measurement-time 10
```

## 🧰 Devcontainer

A ready-to-use devcontainer is provided at `.devcontainer/devcontainer.json` for reproducible development with VS Code or compatible editors.

Requirements:

- Docker (or a compatible container runtime)
- VS Code with the “Dev Containers” extension (or an equivalent)

Usage:

1. Open the project folder in VS Code.
2. When prompted, choose “Reopen in Container” (or use the Command Palette: “Dev Containers: Reopen in Container”).
3. The container installs Rust stable, rustfmt, clippy, llvm-tools, and utilities like cargo-tarpaulin and cargo-criterion.

Common commands inside the devcontainer:

```bash
# Run unit + integration tests
cargo test
cargo test --test cli_integration
cargo test --test cli_integration -- --ignored

# Lints/format
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check

# Benchmarks (Criterion)
cargo bench --bench decoding -- --warm-up-time 0.5 --measurement-time 10
```

Notes:

- Cargo registries are cached via container volumes for faster builds.
- The environment enables colored output and backtraces by default.

## 🐳 Docker Usage

A minimal Dockerfile is included for reproducible builds and tests.

Build the image:

```bash
docker build -t ds-r1-rs:latest .
```

Run the CLI:

```bash
# Print version
docker run --rm ds-r1-rs:latest version

# Show default config
docker run --rm ds-r1-rs:latest config

# Generate text
docker run --rm ds-r1-rs:latest generate "Explain Rust ownership"
```

Mount the project and run from source (optional):

```bash
# Evaluate and export JSON results using the container runtime
docker run --rm -v "$PWD":/work -w /work ds-r1-rs:latest ds-r1-rs eval --json
```

Run tests inside the container:

```bash
# Use the built toolchain to run tests against your mounted workspace
docker run --rm -v "$PWD":/work -w /work ds-r1-rs:latest bash -lc "cargo test --all --release --locked"
```

Tip:

- For faster local iteration, you can keep the container warm and re-run commands without rebuilding the image unless dependencies change.

## 🏗️ Project Structure

```

├── ds-r1_rs/                 # Rust crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs           # CLI: config/version/test/generate/eval
│   │   ├── lib.rs            # Public crate API & re-exports
│   │   ├── model/            # Core model components
│   │   │   ├── config.rs     # Model configuration & validation
│   │   │   ├── transformer.rs# Transformer stack + LM head (implemented)
│   │   │   ├── attention.rs  # Standard attention + MLA + Linear
│   │   │   ├── layers.rs     # Pre-norm TransformerLayer, FFN (SwiGLU), LayerNorm
│   │   │   ├── embeddings.rs # Token + Rotary embeddings
│   │   │   └── moe.rs        # Mixture of Experts (router, experts, load balancing)
│   │   ├── inference/        # Inference & reasoning
│   │   │   ├── engine.rs     # InferenceEngine + high-level solve/explain APIs
│   │   │   ├── generation.rs # Text generation & configs (KV cache placeholder)
│   │   │   ├── sampling.rs   # Greedy/temperature/top-k sampling
│   │   │   ├── reasoning.rs  # <think> parsing, states, analysis
│   │   │   ├── math_solver.rs# Structured math solver utilities
│   │   │   └── code_analyzer.rs
│   │   ├── training/         # Training infrastructure (supervised + RL scaffolding)
│   │   │   ├── data.rs       # Datasets + loaders + synthetic generator
│   │   │   ├── loss.rs       # CrossEntropy + metrics
│   │   │   ├── optimizer.rs  # Adam optimizer
│   │   │   └── trainer.rs    # BasicTrainer + RLTrainer (REINFORCE scaffolding)
│   │   └── utils/            # Errors, math, tokenizer, evaluation harness
│   └── examples/             # Ready-to-run demos
│       ├── generation_demo.rs
│       ├── math_solver_demo.rs
│       ├── training_demo.rs
│       └── config_demo.rs
    └── .github/workflows/ci.yml  # CI: build, lint, test, examples, coverage
```

## 🧩 What’s Implemented

- Model
  - Token embeddings (+ scaling), Rotary embedding (RoPE)
  - Transformer layers with pre-norm and residuals
  - Standard multi-head attention with causal masking
  - Feed-forward with SwiGLU activation
  - Final layer norm + LM head (Linear)
  - Forward pass returning flattened logits `[seq_len * vocab_size]`
- Advanced Modules (standalone, tested)
  - MLA (Multi-head Latent Attention) with compressed KV via LoRA-style compression
  - Mixture of Experts (experts, router, load balancer)
- Inference & Reasoning
  - InferenceEngine with text generation APIs
  - Reasoning-aware generation with <think>…</think> support
  - Reasoning chain parsing, analysis, and structured outputs
- Evaluation
  - EvaluationHarness to run curated benchmarks (math, logic, programming, science, general)
  - Per-problem metrics, performance placeholders, category & difficulty breakdowns
- Training (Prototype)
  - Basic supervised training scaffold (cross-entropy)
  - RL training scaffold (REINFORCE with a simple reward function)
- Utilities
  - Tokenizer (simple, prototype), math helpers, error handling (thiserror)
- Engineering
  - Unit tests across modules
  - CI (fmt, clippy, build, test, run examples, coverage with tarpaulin)
  - Examples showing end-to-end flows

## 🧪 Usage Examples

Programmatic usage:

```rust
use ds_r1_rs::{
    model::{ModelConfig, DeepSeekR1Model},
    inference::engine::InferenceEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build model with validated config
    let config = ModelConfig::default();
    let model = DeepSeekR1Model::new(config)?;

    // Inference engine with default generation configs
    let mut engine = InferenceEngine::new(model)?;

    // Basic generation
    let text = engine.generate_text("The Rust language is")?;
    println!("Generated: {}", text);

    // Reasoning-aware generation
    let reasoning = engine.generate_with_reasoning("Explain ownership in Rust")?;
    println!("Steps: {:?}", reasoning.thinking_chain);
    println!("Answer: {}", reasoning.final_answer);

    Ok(())
}
```

CLI usage (quick):

```bash
cargo run -- config
cargo run -- generate "List 3 benefits of static typing"
cargo run -- eval
cargo run -- eval --json > results.json
cargo run -- save-weights ckpt.json
cargo run -- load-weights ckpt.json "Explain Rust ownership"
```

## 💾 Checkpointing & Reproducibility

You can save/load checkpoints in JSON v1 format. Partial save/load is supported via name-prefix filters. For size-conscious artifacts, use the demo-small model configuration.

Examples:

```bash
# Full save
cargo run -- save-weights ckpt.json

# Partial save/load only lm_head.*
cargo run -- save-weights ckpt.json --include lm_head
cargo run -- load-weights ckpt.json --allow-missing --include lm_head "Your prompt"

# Size-conscious small artifact
cargo run -- save-weights ckpt-small.json --demo-small
cargo run -- load-weights ckpt-small.json --demo-small "Your prompt"

# Deterministic generation (temperature=0 applied automatically in load-weights flow)
cargo run -- load-weights ckpt.json "Explain Rust ownership"
```

## 🧠 How Reasoning Works Here

This prototype uses special thinking tokens and a reasoning state machine to parse and structure “thoughts” during generation:

- The generator can produce `<think> ... </think>` sections.
- The `ReasoningEngine` tracks states (Normal/Thinking/Answering), captures steps, and produces a `ReasoningOutput`.
- The `EvaluationHarness` aggregates metrics (accuracy, clarity, verification presence) across curated benchmarks and reports performance and breakdowns.

## ⚙️ Implementation Notes

- The transformer forward is implemented and functional:
  - Embeddings → N × TransformerLayer → FinalNorm → LM Head
  - Standard attention uses RoPE and causal masking.
  - Output shape is flattened `[seq_len * vocab_size]` for simple integration with training and demos.
- MLA and MoE are fully implemented & unit-tested as components. The main transformer currently uses StandardAttention. Integrating MLA/MoE at scale is left as a roadmap task (see below).
- Generation includes sampling strategies (greedy, temperature, top-k). KV-cache is a placeholder for future optimization.
- Training code is intentionally conservative—scaffolding and examples demonstrate APIs, not production SGD for large checkpoints.

## 🔬 Benchmarks & Evaluation

Use:

```bash
cargo run -- eval
```

This runs curated reasoning benchmarks via the `EvaluationHarness`:

- Mathematics (arithmetic, algebra, word problems, equations)
- Logical reasoning
- Programming logic
- Science reasoning
- General reasoning

Metrics reported:

- Accuracy proxy with numeric tolerance for math answers
- Reasoning depth, clarity, verification presence
- Tokens/sec and reasoning overhead

## 🧭 Roadmap

- Architecture
  - Integrate MLA and MoE into the core transformer stack (selective layer replacement)
  - Add residual/adapter configurations for MLA pathways
  - KV cache for fast autoregressive decoding
  - Configurable layer dropouts, norms, and activation variants
- Inference
  - True streaming token-by-token callback
  - Beam search, top-p sampling
- Training
  - Proper parameter registry and gradient updates across actual model weights
  - Mixed precision, sharding, and large-batch pipelines
- Tooling
  - Load/save checkpoints (weights), export formats
  - More robust tokenizer (BPE/WordPiece)
- Evaluation
  - Richer metrics (exact-match, numeric tolerance for math)
  - Pluggable benchmarks and custom result reporters

## 🧰 CI/CD

GitHub Actions workflow runs on PRs and main:

- rustfmt, clippy (-D warnings)
- build + test all targets
- run all examples
- coverage report via tarpaulin (uploaded as artifact)

## 🤝 Contributing

This is a research/education project. Issues and PRs are welcome. Please:

- Keep code modular, documented, and tested
- Maintain CI green (fmt, clippy, tests)
- Include examples or docs for new features

## 📄 License

MIT — see the crate manifest for details.

Made with insistence by Khaled.
