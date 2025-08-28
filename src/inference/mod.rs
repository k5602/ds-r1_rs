//! # Inference Module
//!
//! Text generation and reasoning inference capabilities.

pub mod code_analyzer;
pub mod engine;
pub mod generation;
pub mod math_solver;
pub mod reasoning;
pub mod sampling;

// Re-export key types
pub use code_analyzer::{CodeAnalyzer, CodeAnalysis, CodeExplanation, CodePattern, Complexity};
pub use engine::InferenceEngine;
pub use generation::{GenerationConfig, GenerationOutput};
pub use math_solver::{MathProblemSolver, MathSolution, MathStep, MathProblemType};
pub use reasoning::{ReasoningEngine, ReasoningOutput, ReasoningState};
pub use sampling::{Sampler, SamplingConfig};
