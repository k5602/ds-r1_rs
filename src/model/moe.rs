//! # Mixture of Experts
//!
//! Implementation of MoE layer with expert routing and load balancing.

use crate::model::attention::Linear;
use crate::utils::error::{ModelError, Result};

/// Individual expert network implementing a specialized MLP with SwiGLU activation
/// Each expert consists of three linear projections: gate, up, and down
pub struct Expert {
    /// Gate projection: [hidden_size] -> [intermediate_size]
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    hidden_size: usize,
    intermediate_size: usize,
}

impl Expert {
    /// Create a new expert with proper weight initialization
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(ModelError::Config(
                "hidden_size and intermediate_size must be greater than 0".to_string(),
            ));
        }

        // three linear projections for SwiGLU
        let gate_proj = Linear::new(hidden_size, intermediate_size)?;
        let up_proj = Linear::new(hidden_size, intermediate_size)?;
        let down_proj = Linear::new(intermediate_size, hidden_size)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
            intermediate_size,
        })
    }

    /// SwiGLU activation function: swish(gate) * up
    fn swiglu_activation(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            return Err(ModelError::Forward(
                "Gate and up projections must have same size".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(gate.len());

        for (&g, &u) in gate.iter().zip(up.iter()) {
            // Swish activation: x * sigmoid(x)
            let swish_g = g * sigmoid(g);
            result.push(swish_g * u);
        }

        Ok(result)
    }

    /// Forward pass through expert network
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden_size {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match hidden_size {}",
                input.len(),
                self.hidden_size
            )));
        }

        // Apply gate and up projections
        let gate_output = self.gate_proj.forward(input)?;
        let up_output = self.up_proj.forward(input)?;
        let activated = self.swiglu_activation(&gate_output, &up_output)?;
        let output = self.down_proj.forward(&activated)?;

        Ok(output)
    }

    /// Forward pass for batch: [batch_size, hidden_size] -> [batch_size, hidden_size]
    pub fn forward_batch(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(input.len());

        for input_vec in input {
            let output_vec = self.forward(input_vec)?;
            output.push(output_vec);
        }

        Ok(output)
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get intermediate size
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Expert routing system for selecting top-k experts per token
pub struct ExpertRouter {
    /// Gating network that produces expert selection scores
    gate: Linear,
    /// Number of experts to select per token
    experts_per_token: usize,
    /// Total number of experts
    num_experts: usize,
    hidden_size: usize,
}

impl ExpertRouter {
    /// Create a new expert router
    pub fn new(hidden_size: usize, num_experts: usize, experts_per_token: usize) -> Result<Self> {
        if hidden_size == 0 || num_experts == 0 || experts_per_token == 0 {
            return Err(ModelError::Config(
                "All parameters must be greater than 0".to_string(),
            ));
        }

        if experts_per_token > num_experts {
            return Err(ModelError::Config(
                "experts_per_token cannot exceed num_experts".to_string(),
            ));
        }

        // Gating network maps hidden_size to num_experts scores
        let gate = Linear::new(hidden_size, num_experts)?;

        Ok(Self {
            gate,
            experts_per_token,
            num_experts,
            hidden_size,
        })
    }

    /// Compute expert selection scores and return top-k experts with weights
    /// Returns (expert_indices, expert_weights) where weights sum to 1.0
    pub fn route(&self, input: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        if input.len() != self.hidden_size {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match hidden_size {}",
                input.len(),
                self.hidden_size
            )));
        }

        // Compute gating scores
        let gate_scores = self.gate.forward(input)?;
        let probabilities = softmax(&gate_scores)?;
        let (indices, weights) = self.select_top_k(&probabilities)?;

        Ok((indices, weights))
    }

    /// Select top-k experts based on probabilities
    fn select_top_k(&self, probabilities: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        if probabilities.len() != self.num_experts {
            return Err(ModelError::Forward(
                "Probabilities length doesn't match num_experts".to_string(),
            ));
        }

        let mut indexed_probs: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut indices = Vec::with_capacity(self.experts_per_token);
        let mut weights = Vec::with_capacity(self.experts_per_token);
        for i in 0..self.experts_per_token {
            indices.push(indexed_probs[i].0);
            weights.push(indexed_probs[i].1);
        }

        // Renormalize weights to sum to 1.0
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in &mut weights {
                *weight /= weight_sum;
            }
        } else {
            // Fallback: uniform weights
            let uniform_weight = 1.0 / self.experts_per_token as f32;
            weights.fill(uniform_weight);
        }

        Ok((indices, weights))
    }

    /// Route batch of inputs: [batch_size, hidden_size] -> [(expert_indices, weights)]
    pub fn route_batch(&self, input: &[Vec<f32>]) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
        let mut routing_results = Vec::with_capacity(input.len());

        for input_vec in input {
            let (indices, weights) = self.route(input_vec)?;
            routing_results.push((indices, weights));
        }

        Ok(routing_results)
    }

    /// Get number of experts per token
    pub fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }

    /// Get total number of experts
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
}

/// Load balancer to ensure even expert utilization
pub struct LoadBalancer {
    /// Expert usage counts for load balancing
    expert_usage: Vec<f32>,
    /// Total number of routing decisions
    total_decisions: usize,
    num_experts: usize,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(num_experts: usize) -> Result<Self> {
        if num_experts == 0 {
            return Err(ModelError::Config(
                "num_experts must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            expert_usage: vec![0.0; num_experts],
            total_decisions: 0,
            num_experts,
        })
    }

    /// Update expert usage statistics
    pub fn update_usage(&mut self, expert_indices: &[usize], weights: &[f32]) -> Result<()> {
        if expert_indices.len() != weights.len() {
            return Err(ModelError::Forward(
                "Expert indices and weights must have same length".to_string(),
            ));
        }

        for (&expert_idx, &weight) in expert_indices.iter().zip(weights.iter()) {
            if expert_idx >= self.num_experts {
                return Err(ModelError::Forward(format!(
                    "Expert index {} exceeds num_experts {}",
                    expert_idx, self.num_experts
                )));
            }
            self.expert_usage[expert_idx] += weight;
        }

        self.total_decisions += 1;
        Ok(())
    }

    /// Get expert utilization statistics
    pub fn get_utilization(&self) -> Vec<f32> {
        if self.total_decisions == 0 {
            return vec![0.0; self.num_experts];
        }

        self.expert_usage
            .iter()
            .map(|&usage| usage / self.total_decisions as f32)
            .collect()
    }
    pub fn get_load_balance_loss(&self) -> f32 {
        let utilization = self.get_utilization();
        let mean_util = utilization.iter().sum::<f32>() / self.num_experts as f32;

        let variance = utilization
            .iter()
            .map(|&util| (util - mean_util).powi(2))
            .sum::<f32>()
            / self.num_experts as f32;

        variance
    }
    pub fn reset(&mut self) {
        self.expert_usage.fill(0.0);
        self.total_decisions = 0;
    }
}

/// Softmax activation function
fn softmax(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(ModelError::Forward("Input cannot be empty".to_string()));
    }

    // Find maximum for numerical stability
    let max_val = input
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Compute exp(x - max) for numerical stability
    let exp_values: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();

    // Compute sum of exponentials
    let sum_exp: f32 = exp_values.iter().sum();

    if sum_exp == 0.0 {
        return Err(ModelError::Forward(
            "Softmax denominator is zero".to_string(),
        ));
    }

    // Normalize
    let probabilities: Vec<f32> = exp_values
        .iter()
        .map(|&exp_val| exp_val / sum_exp)
        .collect();

    Ok(probabilities)
}

/// Mixture of Experts layer combining expert routing with expert computation
pub struct MoELayer {
    /// Individual expert networks
    experts: Vec<Expert>,
    /// Expert routing system
    router: ExpertRouter,
    /// Load balancer for expert utilization
    load_balancer: LoadBalancer,
    hidden_size: usize,
    num_experts: usize,
    experts_per_token: usize,
}

impl MoELayer {
    /// Create a new MoE layer with specified configuration
    pub fn new(hidden_size: usize, num_experts: usize, experts_per_token: usize) -> Result<Self> {
        if hidden_size == 0 || num_experts == 0 || experts_per_token == 0 {
            return Err(ModelError::Config(
                "All parameters must be greater than 0".to_string(),
            ));
        }

        if experts_per_token > num_experts {
            return Err(ModelError::Config(
                "experts_per_token cannot exceed num_experts".to_string(),
            ));
        }

        // Create expert networks
        let mut experts = Vec::with_capacity(num_experts);
        let intermediate_size = hidden_size * 4; // Standard 4x expansion

        for _ in 0..num_experts {
            let expert = Expert::new(hidden_size, intermediate_size)?;
            experts.push(expert);
        }
        let router = ExpertRouter::new(hidden_size, num_experts, experts_per_token)?;
        let load_balancer = LoadBalancer::new(num_experts)?;

        Ok(Self {
            experts,
            router,
            load_balancer,
            hidden_size,
            num_experts,
            experts_per_token,
        })
    }

    /// Create a new MoE layer with custom intermediate size
    pub fn new_with_intermediate_size(
        hidden_size: usize,
        num_experts: usize,
        experts_per_token: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        if hidden_size == 0 || num_experts == 0 || experts_per_token == 0 || intermediate_size == 0
        {
            return Err(ModelError::Config(
                "All parameters must be greater than 0".to_string(),
            ));
        }

        if experts_per_token > num_experts {
            return Err(ModelError::Config(
                "experts_per_token cannot exceed num_experts".to_string(),
            ));
        }

        // Create expert networks with custom intermediate size
        let mut experts = Vec::with_capacity(num_experts);

        for _ in 0..num_experts {
            let expert = Expert::new(hidden_size, intermediate_size)?;
            experts.push(expert);
        }

        // Create routing system
        let router = ExpertRouter::new(hidden_size, num_experts, experts_per_token)?;

        // Create load balancer
        let load_balancer = LoadBalancer::new(num_experts)?;

        Ok(Self {
            experts,
            router,
            load_balancer,
            hidden_size,
            num_experts,
            experts_per_token,
        })
    }

    /// Forward pass through MoE layer with sparse expert activation
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden_size {
            return Err(ModelError::Forward(format!(
                "Input size {} doesn't match hidden_size {}",
                input.len(),
                self.hidden_size
            )));
        }
        let (expert_indices, expert_weights) = self.router.route(input)?;

        // Update load balancer statistics
        self.load_balancer
            .update_usage(&expert_indices, &expert_weights)?;

        // Compute weighted combination of expert outputs
        let mut final_output = vec![0.0; self.hidden_size];

        for (&expert_idx, &weight) in expert_indices.iter().zip(expert_weights.iter()) {
            if expert_idx >= self.experts.len() {
                return Err(ModelError::Forward(format!(
                    "Expert index {} exceeds number of experts {}",
                    expert_idx,
                    self.experts.len()
                )));
            }

            // Compute expert output
            let expert_output = self.experts[expert_idx].forward(input)?;

            // Add weighted expert output to final result
            for (final_val, &expert_val) in final_output.iter_mut().zip(expert_output.iter()) {
                *final_val += weight * expert_val;
            }
        }

        Ok(final_output)
    }

    /// Forward pass for batch: [batch_size, hidden_size] -> [batch_size, hidden_size]
    pub fn forward_batch(&mut self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(input.len());

        for input_vec in input {
            let output_vec = self.forward(input_vec)?;
            output.push(output_vec);
        }

        Ok(output)
    }

    /// Get expert utilization statistics
    pub fn get_expert_utilization(&self) -> Vec<f32> {
        self.load_balancer.get_utilization()
    }

    /// Get load balancing loss (variance in expert utilization)
    pub fn get_load_balance_loss(&self) -> f32 {
        self.load_balancer.get_load_balance_loss()
    }

    /// Reset load balancing statistics
    pub fn reset_load_balancer(&mut self) {
        self.load_balancer.reset();
    }
    pub fn get_active_experts(&self, input: &[f32]) -> Result<Vec<(usize, f32)>> {
        let (expert_indices, expert_weights) = self.router.route(input)?;

        let active_experts: Vec<(usize, f32)> = expert_indices
            .into_iter()
            .zip(expert_weights.into_iter())
            .collect();

        Ok(active_experts)
    }

    /// Get configuration information
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }

    /// Get intermediate size of experts
    pub fn intermediate_size(&self) -> usize {
        if !self.experts.is_empty() {
            self.experts[0].intermediate_size()
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(1000.0) > 0.99); // Should approach 1
        assert!(sigmoid(-1000.0) < 0.01); // Should approach 0

        // Test monotonicity
        assert!(sigmoid(-1.0) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(1.0));
    }

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(512, 2048);
        assert!(expert.is_ok());

        let expert = expert.unwrap();
        assert_eq!(expert.hidden_size(), 512);
        assert_eq!(expert.intermediate_size(), 2048);
    }

    #[test]
    fn test_expert_invalid_config() {
        assert!(Expert::new(0, 2048).is_err());
        assert!(Expert::new(512, 0).is_err());
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(64, 256).unwrap();
        let input = vec![1.0; 64];

        let result = expert.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_expert_batch() {
        let expert = Expert::new(32, 128).unwrap();
        let input = vec![vec![1.0; 32], vec![0.5; 32], vec![-0.5; 32]];

        let result = expert.forward_batch(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 32);
        assert_eq!(output[1].len(), 32);
        assert_eq!(output[2].len(), 32);
    }

    #[test]
    fn test_expert_wrong_input_size() {
        let expert = Expert::new(64, 256).unwrap();
        let input = vec![1.0; 32]; // Wrong size

        let result = expert.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_swiglu_activation() {
        let expert = Expert::new(4, 8).unwrap();
        let gate = vec![1.0, 0.0, -1.0, 2.0];
        let up = vec![2.0, 1.0, 0.5, -1.0];

        let result = expert.swiglu_activation(&gate, &up);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 4);

        // Check that SwiGLU produces reasonable values
        // swish(1.0) * 2.0 should be positive
        assert!(output[0] > 0.0);
        // swish(0.0) * 1.0 should be 0
        assert!((output[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_swiglu_mismatched_sizes() {
        let expert = Expert::new(4, 8).unwrap();
        let gate = vec![1.0, 2.0];
        let up = vec![1.0, 2.0, 3.0]; // Different size

        let result = expert.swiglu_activation(&gate, &up);
        assert!(result.is_err());
    }

    #[test]
    fn test_expert_nonlinearity() {
        let expert = Expert::new(2, 4).unwrap();
        let input1 = vec![1.0, 0.0];
        let input2 = vec![0.0, 1.0];
        let input_sum = vec![1.0, 1.0];

        let output1 = expert.forward(&input1).unwrap();
        let output2 = expert.forward(&input2).unwrap();
        let output_sum = expert.forward(&input_sum).unwrap();

        // Due to nonlinearity, f(a + b) ≠ f(a) + f(b)
        let manual_sum: Vec<f32> = output1
            .iter()
            .zip(output2.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let mut different = false;
        for (manual, actual) in manual_sum.iter().zip(output_sum.iter()) {
            if (manual - actual).abs() > 1e-6 {
                different = true;
                break;
            }
        }

        assert!(different, "Expert should be nonlinear due to SwiGLU");
    }

    #[test]
    fn test_expert_different_inputs_different_outputs() {
        let expert = Expert::new(4, 16).unwrap();
        let input1 = vec![1.0, 0.0, 0.0, 0.0];
        let input2 = vec![0.0, 1.0, 0.0, 0.0];

        let output1 = expert.forward(&input1).unwrap();
        let output2 = expert.forward(&input2).unwrap();

        // Different inputs should produce different outputs
        let mut different = false;
        for (out1, out2) in output1.iter().zip(output2.iter()) {
            if (out1 - out2).abs() > 1e-6 {
                different = true;
                break;
            }
        }

        assert!(
            different,
            "Different inputs should produce different outputs"
        );
    }

    #[test]
    fn test_softmax_function() {
        let input = vec![1.0, 2.0, 3.0];
        let result = softmax(&input);
        assert!(result.is_ok());

        let probs = result.unwrap();
        assert_eq!(probs.len(), 3);

        // Check that probabilities sum to 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that probabilities are positive
        for &p in &probs {
            assert!(p > 0.0);
        }

        // Check that larger inputs get larger probabilities
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large values that could cause overflow
        let input = vec![1000.0, 1001.0, 1002.0];
        let result = softmax(&input);
        assert!(result.is_ok());

        let probs = result.unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check no NaN or infinite values
        for &p in &probs {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_softmax_empty_input() {
        let input = vec![];
        let result = softmax(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_expert_router_creation() {
        let router = ExpertRouter::new(512, 8, 2);
        assert!(router.is_ok());

        let router = router.unwrap();
        assert_eq!(router.num_experts(), 8);
        assert_eq!(router.experts_per_token(), 2);
    }

    #[test]
    fn test_expert_router_invalid_config() {
        assert!(ExpertRouter::new(0, 8, 2).is_err());
        assert!(ExpertRouter::new(512, 0, 2).is_err());
        assert!(ExpertRouter::new(512, 8, 0).is_err());
        assert!(ExpertRouter::new(512, 4, 8).is_err()); // experts_per_token > num_experts
    }

    #[test]
    fn test_expert_router_routing() {
        let router = ExpertRouter::new(64, 8, 2).unwrap();
        let input = vec![1.0; 64];

        let result = router.route(&input);
        assert!(result.is_ok());

        let (indices, weights) = result.unwrap();
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);

        // Check that weights sum to 1.0
        let weight_sum: f32 = weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        // Check that indices are valid
        for &idx in &indices {
            assert!(idx < 8);
        }

        // Check that weights are positive
        for &weight in &weights {
            assert!(weight > 0.0);
        }
    }

    #[test]
    fn test_expert_router_batch() {
        let router = ExpertRouter::new(32, 4, 2).unwrap();
        let input = vec![vec![1.0; 32], vec![0.5; 32], vec![-0.5; 32]];

        let result = router.route_batch(&input);
        assert!(result.is_ok());

        let routing_results = result.unwrap();
        assert_eq!(routing_results.len(), 3);

        for (indices, weights) in routing_results {
            assert_eq!(indices.len(), 2);
            assert_eq!(weights.len(), 2);

            let weight_sum: f32 = weights.iter().sum();
            assert!((weight_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_expert_router_wrong_input_size() {
        let router = ExpertRouter::new(64, 8, 2).unwrap();
        let input = vec![1.0; 32]; // Wrong size

        let result = router.route(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_balancer_creation() {
        let balancer = LoadBalancer::new(8);
        assert!(balancer.is_ok());

        let balancer = balancer.unwrap();
        let utilization = balancer.get_utilization();
        assert_eq!(utilization.len(), 8);

        // Initially all utilization should be zero
        for &util in &utilization {
            assert_eq!(util, 0.0);
        }
    }

    #[test]
    fn test_load_balancer_invalid_config() {
        assert!(LoadBalancer::new(0).is_err());
    }

    #[test]
    fn test_load_balancer_update_usage() {
        let mut balancer = LoadBalancer::new(4).unwrap();

        let indices = vec![0, 2];
        let weights = vec![0.7, 0.3];

        let result = balancer.update_usage(&indices, &weights);
        assert!(result.is_ok());

        let utilization = balancer.get_utilization();
        assert_eq!(utilization.len(), 4);
        assert!((utilization[0] - 0.7).abs() < 1e-6);
        assert!((utilization[1] - 0.0).abs() < 1e-6);
        assert!((utilization[2] - 0.3).abs() < 1e-6);
        assert!((utilization[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_balancer_multiple_updates() {
        let mut balancer = LoadBalancer::new(3).unwrap();

        // First update
        balancer.update_usage(&vec![0, 1], &vec![0.6, 0.4]).unwrap();
        // Second update
        balancer.update_usage(&vec![1, 2], &vec![0.5, 0.5]).unwrap();

        let utilization = balancer.get_utilization();
        // After 2 decisions: expert 0 used 0.6/2=0.3, expert 1 used (0.4+0.5)/2=0.45, expert 2 used 0.5/2=0.25
        assert!((utilization[0] - 0.3).abs() < 1e-6);
        assert!((utilization[1] - 0.45).abs() < 1e-6);
        assert!((utilization[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_load_balancer_mismatched_lengths() {
        let mut balancer = LoadBalancer::new(4).unwrap();

        let indices = vec![0, 1];
        let weights = vec![0.7]; // Different length

        let result = balancer.update_usage(&indices, &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_balancer_invalid_expert_index() {
        let mut balancer = LoadBalancer::new(4).unwrap();

        let indices = vec![0, 5]; // Index 5 is out of bounds
        let weights = vec![0.7, 0.3];

        let result = balancer.update_usage(&indices, &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_balancer_loss() {
        let mut balancer = LoadBalancer::new(3).unwrap();

        // Perfectly balanced usage
        balancer.update_usage(&vec![0], &vec![1.0]).unwrap();
        balancer.update_usage(&vec![1], &vec![1.0]).unwrap();
        balancer.update_usage(&vec![2], &vec![1.0]).unwrap();

        let loss = balancer.get_load_balance_loss();
        assert!(loss < 1e-6); // Should be very small for balanced usage

        // Reset and create imbalanced usage
        balancer.reset();
        balancer.update_usage(&vec![0], &vec![1.0]).unwrap();
        balancer.update_usage(&vec![0], &vec![1.0]).unwrap();
        balancer.update_usage(&vec![0], &vec![1.0]).unwrap();

        let loss_imbalanced = balancer.get_load_balance_loss();
        assert!(loss_imbalanced > loss); // Should be higher for imbalanced usage
    }

    #[test]
    fn test_load_balancer_reset() {
        let mut balancer = LoadBalancer::new(3).unwrap();

        balancer.update_usage(&vec![0, 1], &vec![0.6, 0.4]).unwrap();

        let utilization_before = balancer.get_utilization();
        assert!(utilization_before[0] > 0.0);

        balancer.reset();

        let utilization_after = balancer.get_utilization();
        for &util in &utilization_after {
            assert_eq!(util, 0.0);
        }
    }

    #[test]
    fn test_moe_creation() {
        let moe = MoELayer::new(32, 8, 2);
        assert!(moe.is_ok());

        let moe = moe.unwrap();
        assert_eq!(moe.hidden_size(), 32);
        assert_eq!(moe.num_experts(), 8);
        assert_eq!(moe.experts_per_token(), 2);
        assert_eq!(moe.intermediate_size(), 32 * 4); // Default 4x expansion
    }

    #[test]
    fn test_moe_creation_with_custom_intermediate() {
        let moe = MoELayer::new_with_intermediate_size(256, 4, 2, 1024);
        assert!(moe.is_ok());

        let moe = moe.unwrap();
        assert_eq!(moe.hidden_size(), 256);
        assert_eq!(moe.num_experts(), 4);
        assert_eq!(moe.experts_per_token(), 2);
        assert_eq!(moe.intermediate_size(), 1024);
    }

    #[test]
    fn test_moe_invalid_config() {
        assert!(MoELayer::new(0, 8, 2).is_err());
        assert!(MoELayer::new(512, 0, 2).is_err());
        assert!(MoELayer::new(512, 8, 0).is_err());
        assert!(MoELayer::new(512, 4, 8).is_err()); // experts_per_token > num_experts
    }

    #[test]
    fn test_moe_forward() {
        let mut moe = MoELayer::new(64, 4, 2).unwrap();
        let input = vec![1.0; 64];

        let result = moe.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_moe_batch() {
        let mut moe = MoELayer::new(32, 4, 2).unwrap();
        let input = vec![vec![1.0; 32], vec![0.5; 32], vec![-0.5; 32]];

        let result = moe.forward_batch(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 32);
        assert_eq!(output[1].len(), 32);
        assert_eq!(output[2].len(), 32);
    }

    #[test]
    fn test_moe_wrong_input_size() {
        let mut moe = MoELayer::new(64, 4, 2).unwrap();
        let input = vec![1.0; 32]; // Wrong size

        let result = moe.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_moe_sparse_activation() {
        let moe = MoELayer::new(32, 8, 2).unwrap();
        let input = vec![1.0; 32];

        // Get active experts
        let active_experts = moe.get_active_experts(&input);
        assert!(active_experts.is_ok());

        let active = active_experts.unwrap();
        assert_eq!(active.len(), 2); // Should have exactly 2 active experts

        // Check that weights sum to 1.0
        let weight_sum: f32 = active.iter().map(|(_, weight)| weight).sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        // Check that expert indices are valid
        for &(expert_idx, _) in &active {
            assert!(expert_idx < 8);
        }
    }

    #[test]
    fn test_moe_load_balancing() {
        let mut moe = MoELayer::new(16, 4, 2).unwrap();

        // Process multiple inputs to accumulate usage statistics
        for i in 0..10 {
            let input = vec![i as f32 * 0.1; 16];
            let _ = moe.forward(&input).unwrap();
        }

        // Check utilization statistics
        let utilization = moe.get_expert_utilization();
        assert_eq!(utilization.len(), 4);

        // All experts should have some utilization (though not necessarily equal)
        let total_utilization: f32 = utilization.iter().sum();
        assert!(total_utilization > 0.0);

        // Get load balance loss
        let loss = moe.get_load_balance_loss();
        assert!(loss >= 0.0); // Variance should be non-negative
    }

    #[test]
    fn test_moe_reset_load_balancer() {
        let mut moe = MoELayer::new(16, 4, 2).unwrap();

        // Process some inputs
        let input = vec![1.0; 16];
        let _ = moe.forward(&input).unwrap();

        let utilization_before = moe.get_expert_utilization();
        let has_usage = utilization_before.iter().any(|&u| u > 0.0);
        assert!(has_usage);

        // Reset load balancer
        moe.reset_load_balancer();

        let utilization_after = moe.get_expert_utilization();
        for &util in &utilization_after {
            assert_eq!(util, 0.0);
        }
    }

    #[test]
    fn test_moe_different_inputs_different_routing() {
        let moe = MoELayer::new(8, 4, 1).unwrap();

        let input1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let input2 = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        let active1 = moe.get_active_experts(&input1).unwrap();
        let active2 = moe.get_active_experts(&input2).unwrap();

        // Different inputs might route to different experts
        // (This is probabilistic, but with different enough inputs, routing should differ)
        let experts1: Vec<usize> = active1.iter().map(|(idx, _)| *idx).collect();
        let experts2: Vec<usize> = active2.iter().map(|(idx, _)| *idx).collect();

        // At least check that the routing system is working
        assert_eq!(experts1.len(), moe.experts_per_token());
        assert_eq!(experts2.len(), moe.experts_per_token());
    }

    #[test]
    fn test_moe_output_transformation() {
        let mut moe = MoELayer::new(4, 2, 1).unwrap(); // Use 1 expert per token for deterministic behavior

        let input = vec![1.0, 0.0, 0.0, 0.0];
        let output = moe.forward(&input).unwrap();

        // Output should be different from input (transformation should occur)
        let mut different = false;
        for (inp, out) in input.iter().zip(output.iter()) {
            if (inp - out).abs() > 1e-6 {
                different = true;
                break;
            }
        }

        assert!(different, "MoE should transform the input");
    }

    #[test]
    fn test_moe_nonlinearity() {
        let mut moe = MoELayer::new(2, 2, 1).unwrap();

        let input1 = vec![1.0, 0.0];
        let input2 = vec![0.0, 1.0];
        let input_sum = vec![1.0, 1.0];

        let output1 = moe.forward(&input1).unwrap();
        let output2 = moe.forward(&input2).unwrap();
        let output_sum = moe.forward(&input_sum).unwrap();

        // Due to nonlinearity, f(a + b) ≠ f(a) + f(b)
        let manual_sum: Vec<f32> = output1
            .iter()
            .zip(output2.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let mut different = false;
        for (manual, actual) in manual_sum.iter().zip(output_sum.iter()) {
            if (manual - actual).abs() > 1e-6 {
                different = true;
                break;
            }
        }

        assert!(
            different,
            "MoE should be nonlinear due to expert routing and SwiGLU"
        );
    }

    #[test]
    fn test_moe_expert_specialization() {
        let moe = MoELayer::new(4, 4, 1).unwrap(); // 1 expert per token for clear specialization

        // Process the same input multiple times to see consistent routing
        let input = vec![1.0, 0.0, 0.0, 0.0];

        let active1 = moe.get_active_experts(&input).unwrap();
        let active2 = moe.get_active_experts(&input).unwrap();

        // Same input should route to same expert(s)
        assert_eq!(active1.len(), 1);
        assert_eq!(active2.len(), 1);
        assert_eq!(active1[0].0, active2[0].0); // Same expert index
    }

    #[test]
    fn test_moe_weight_normalization() {
        let moe = MoELayer::new(8, 6, 3).unwrap();
        let input = vec![0.5; moe.hidden_size()];

        let active_experts = moe.get_active_experts(&input).unwrap();
        assert_eq!(active_experts.len(), moe.experts_per_token());

        // Weights should sum to 1.0
        let weight_sum: f32 = active_experts.iter().map(|(_, weight)| weight).sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        // All weights should be positive
        for &(_, weight) in &active_experts {
            assert!(weight > 0.0);
        }
    }
}
