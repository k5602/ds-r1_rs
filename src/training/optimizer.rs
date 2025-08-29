//! # Optimizers
//!
//! Optimization algorithms for training.

use crate::utils::error::{ModelError, Result};

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Parameter state for Adam optimizer
#[derive(Debug, Clone)]
struct ParameterState {
    momentum: Vec<f32>,
    velocity: Vec<f32>,
}

/// Adam optimizer implementation
pub struct Optimizer {
    config: OptimizerConfig,
    step_count: usize,
    parameter_states: std::collections::HashMap<String, ParameterState>,
}

impl Optimizer {
    /// Create a new Adam optimizer
    pub fn new(config: OptimizerConfig) -> Result<Self> {
        Ok(Self {
            config,
            step_count: 0,
            parameter_states: std::collections::HashMap::new(),
        })
    }

    /// Initialize parameter state if not exists
    fn init_parameter_state(&mut self, param_name: &str, param_size: usize) {
        if !self.parameter_states.contains_key(param_name) {
            self.parameter_states.insert(
                param_name.to_string(),
                ParameterState {
                    momentum: vec![0.0; param_size],
                    velocity: vec![0.0; param_size],
                },
            );
        }
    }

    /// Perform Adam optimization step for a single parameter
    pub fn step_parameter(
        &mut self,
        param_name: &str,
        parameters: &mut [f32],
        gradients: &[f32],
    ) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(ModelError::Training(format!(
                "Parameter and gradient size mismatch: {} vs {}",
                parameters.len(),
                gradients.len()
            )));
        }

        self.init_parameter_state(param_name, parameters.len());
        self.step_count += 1;

        let state = self.parameter_states.get_mut(param_name).unwrap();

        // Bias correction terms
        let bias_correction1 = 1.0 - self.config.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.step_count as i32);

        for i in 0..parameters.len() {
            let grad = gradients[i];

            // Update biased first moment estimate
            state.momentum[i] =
                self.config.beta1 * state.momentum[i] + (1.0 - self.config.beta1) * grad;

            // Update biased second raw moment estimate
            state.velocity[i] =
                self.config.beta2 * state.velocity[i] + (1.0 - self.config.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            let m_hat = state.momentum[i] / bias_correction1;

            // Compute bias-corrected second raw moment estimate
            let v_hat = state.velocity[i] / bias_correction2;

            // Update parameters
            let update = self.config.learning_rate * m_hat / (v_hat.sqrt() + self.config.epsilon);
            parameters[i] -= update + self.config.weight_decay * parameters[i];
        }

        Ok(())
    }

    /// Perform optimization step (legacy interface)
    pub fn step(&mut self, _gradients: &[f32]) -> Result<()> {
        // This is a simplified interface for backward compatibility
        // In practice, we'd use step_parameter for each parameter group
        Ok(())
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.parameter_states.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.beta1 > 0.0 && config.beta1 < 1.0);
        assert!(config.beta2 > 0.0 && config.beta2 < 1.0);
    }

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer = Optimizer::new(config);
        assert!(optimizer.is_ok());
    }
}
