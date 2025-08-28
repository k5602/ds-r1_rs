//! # Training Infrastructure
//!
//! Basic trainer implementations for supervised and reinforcement learning.

use crate::model::DeepSeekR1Model;
use crate::training::data::{TrainingBatch, TrainingExample};
use crate::training::loss::{CrossEntropyLoss, LossFunction, TrainingMetrics};
use crate::training::optimizer::{Optimizer, OptimizerConfig};
use crate::utils::error::{ModelError, Result};

/// Basic supervised trainer
pub struct BasicTrainer {
    model: DeepSeekR1Model,
    optimizer: Optimizer,
    loss_fn: CrossEntropyLoss,
    step_count: usize,
    vocab_size: usize,
}

impl BasicTrainer {
    /// Create a new basic trainer
    pub fn new(model: DeepSeekR1Model) -> Result<Self> {
        let optimizer_config = OptimizerConfig::default();
        let optimizer = Optimizer::new(optimizer_config)?;
        let loss_fn = CrossEntropyLoss::new();
        let vocab_size = model.config().vocab_size;

        Ok(Self {
            model,
            optimizer,
            loss_fn,
            step_count: 0,
            vocab_size,
        })
    }

    /// Create a new basic trainer with custom optimizer config
    pub fn with_optimizer_config(
        model: DeepSeekR1Model,
        optimizer_config: OptimizerConfig,
    ) -> Result<Self> {
        let optimizer = Optimizer::new(optimizer_config)?;
        let loss_fn = CrossEntropyLoss::new();
        let vocab_size = model.config().vocab_size;

        Ok(Self {
            model,
            optimizer,
            loss_fn,
            step_count: 0,
            vocab_size,
        })
    }

    /// Convert text to token IDs using a simple word-based tokenizer
    fn tokenize(&self, text: &str) -> Vec<u32> {
        // Simple word-based tokenization with basic preprocessing
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding.split_whitespace().collect();

        let mut token_ids = Vec::new();

        for word in words {
            // Create a simple hash-based token ID
            let mut hash = 0u32;
            for byte in word.bytes() {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
            }
            // Ensure token ID is within vocabulary range
            token_ids.push(hash % self.vocab_size as u32);
        }

        // Add special tokens if empty
        if token_ids.is_empty() {
            token_ids.push(0); // UNK token
        }

        token_ids
    }

    /// Prepare training data from examples
    fn prepare_training_data(&self, examples: &[TrainingExample]) -> Result<(Vec<u32>, Vec<u32>)> {
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();

        for example in examples {
            // Tokenize input and target
            let input_tokens = self.tokenize(&example.input);
            let target_tokens = self.tokenize(&example.target);

            // For next-token prediction, we shift targets by one position
            input_ids.extend(input_tokens);
            target_ids.extend(target_tokens);
        }

        Ok((input_ids, target_ids))
    }

    /// Prepare last-step training data: per-example input token sequences and single-target class
    fn prepare_last_step_data(
        &self,
        examples: &[TrainingExample],
    ) -> Result<(Vec<Vec<u32>>, Vec<u32>)> {
        let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(examples.len());
        let mut targets: Vec<u32> = Vec::with_capacity(examples.len());

        for example in examples {
            let input_tokens = self.tokenize(&example.input);
            // For a classification-like next-token objective, use the first token of the target string
            let target_tokens = self.tokenize(&example.target);
            let target_id = target_tokens.get(0).copied().unwrap_or(0);

            inputs.push(input_tokens);
            targets.push(target_id);
        }

        Ok((inputs, targets))
    }

    /// Compute gradients using backpropagation through cross-entropy loss
    fn compute_gradients(&self, predictions: &[f32], targets: &[u32]) -> Result<Vec<f32>> {
        let mut gradients = vec![0.0; predictions.len()];
        let vocab_size = self.vocab_size;
        let num_samples = targets.len();

        if predictions.len() != num_samples * vocab_size {
            return Err(ModelError::Training(format!(
                "Prediction size mismatch: expected {}, got {}",
                num_samples * vocab_size,
                predictions.len()
            )));
        }

        for (i, &target) in targets.iter().enumerate() {
            let start_idx = i * vocab_size;
            let end_idx = start_idx + vocab_size;

            if end_idx <= predictions.len() && (target as usize) < vocab_size {
                let logits = &predictions[start_idx..end_idx];

                // Compute softmax probabilities with numerical stability
                let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();

                // Compute gradients of cross-entropy loss w.r.t. logits
                for j in 0..vocab_size {
                    let grad_idx = start_idx + j;
                    let prob = exp_logits[j] / sum_exp;

                    if j == target as usize {
                        // Gradient for correct class: p - 1
                        gradients[grad_idx] = prob - 1.0;
                    } else {
                        // Gradient for incorrect class: p
                        gradients[grad_idx] = prob;
                    }
                }
            }
        }

        // Normalize gradients by batch size
        let batch_size = num_samples as f32;
        for grad in &mut gradients {
            *grad /= batch_size;
        }

        Ok(gradients)
    }

    /// Perform a training step
    pub fn train_step(&mut self, batch: &TrainingBatch) -> Result<TrainingMetrics> {
        if batch.examples.is_empty() {
            return Err(ModelError::Training("Empty batch".to_string()));
        }

        // Prepare per-example last-step data
        let (inputs_per_example, targets) = self.prepare_last_step_data(&batch.examples)?;

        if inputs_per_example.is_empty() || targets.is_empty() {
            return Err(ModelError::Training("No valid training data".to_string()));
        }

        // Forward per-example (last-step) and collect last hiddens and last input ids
        let (predictions, last_hiddens, last_input_ids) =
            self.forward_last_step(&inputs_per_example)?;

        // Compute loss
        let target_floats: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
        let loss = self.loss_fn.compute_loss(&predictions, &target_floats)?;

        // Compute accuracy
        let accuracy = self.loss_fn.compute_accuracy(&predictions, &targets);

        // Compute gradients
        let gradients = self.compute_gradients(&predictions, &targets)?;

        // Update model parameters using computed gradients (LM head weights + bias + embeddings)
        self.update_model_parameters(&gradients, &last_hiddens, &last_input_ids)?;

        self.step_count += 1;

        Ok(TrainingMetrics::new(loss, accuracy, self.step_count))
    }

    /// Forward pass through the model
    fn forward_pass(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        // Use the actual model forward pass
        let logits = self.model.forward(input_ids)?;
        Ok(logits)
    }

    /// Forward per-example and return concatenated last-step logits, last hidden states, and last input ids
    fn forward_last_step(
        &mut self,
        inputs: &[Vec<u32>],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<u32>)> {
        let mut predictions: Vec<f32> = Vec::new();
        let mut last_hiddens: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        let mut last_input_ids: Vec<u32> = Vec::with_capacity(inputs.len());

        for input in inputs {
            if input.is_empty() {
                return Err(ModelError::Training("Empty input sequence".to_string()));
            }

            // Full forward for this example
            let logits = self.model.forward(input)?;
            let vocab_size = self.vocab_size;
            if logits.len() < vocab_size {
                return Err(ModelError::Training(
                    "Model output size doesn't match vocabulary size".to_string(),
                ));
            }
            // Take last position logits only
            let last_logits = &logits[logits.len() - vocab_size..];
            predictions.extend_from_slice(last_logits);

            // Final hidden states and take only last one
            let hidden = self.model.forward_hidden(input)?;
            let last_h = hidden
                .last()
                .ok_or_else(|| ModelError::Training("No hidden states".to_string()))?
                .clone();
            last_hiddens.push(last_h);

            // Track last input id for embedding update
            last_input_ids.push(*input.last().unwrap());
        }

        Ok((predictions, last_hiddens, last_input_ids))
    }

    /// Update model parameters using gradients from CE on logits.
    /// Implements a minimal backward path for the LM head (weights + bias) and token embeddings.
    /// - dW_lm_head = sum_t outer(dlogits_t, h_t)
    /// - db_lm_head = sum_t dlogits_t
    /// - dembed[token_t] += W^T * dlogits_t   (minimal proxy ignoring intermediate layers)
    fn update_model_parameters(
        &mut self,
        gradients: &[f32],
        hidden: &[Vec<f32>],
        input_ids: &[u32],
    ) -> Result<()> {
        let vocab_size = self.vocab_size;
        if vocab_size == 0 {
            return Err(ModelError::Training("Vocab size is zero".to_string()));
        }
        if gradients.len() % vocab_size != 0 {
            return Err(ModelError::Training(format!(
                "Gradients length {} is not divisible by vocab_size {}",
                gradients.len(),
                vocab_size
            )));
        }
        let num_samples = gradients.len() / vocab_size;
        if hidden.len() != num_samples || input_ids.len() != num_samples {
            return Err(ModelError::Training(format!(
                "Mismatch: hidden len {} / input_ids len {} vs samples {}",
                hidden.len(),
                input_ids.len(),
                num_samples
            )));
        }

        // 1) Bias gradients: db = sum_t dlogits_t
        let mut bias_grads = vec![0.0f32; vocab_size];
        for (i, chunk) in gradients.chunks(vocab_size).enumerate() {
            let _ = i; // unused
            for v in 0..vocab_size {
                bias_grads[v] += chunk[v];
            }
        }

        // Apply bias update
        {
            let name = "lm_head.bias";
            let bias_slice = self.model.lm_head_bias_mut();
            self.optimizer
                .step_parameter(name, bias_slice, &bias_grads)?;
        }

        // 2) Weight gradients: dW[v] = sum_t dlogits_t[v] * h_t
        // Accumulate per row to avoid storing full matrix if not needed.
        // Use hidden_size inferred from a row of lm_head.
        let lm_w_snapshot = self.model.lm_head_weights().clone();
        if lm_w_snapshot.is_empty() {
            return Err(ModelError::Training(
                "LM head weights are empty".to_string(),
            ));
        }
        let hidden_size = lm_w_snapshot[0].len();

        for v in 0..vocab_size {
            let mut row_grad = vec![0.0f32; hidden_size];
            for t in 0..num_samples {
                let g_vt = gradients[t * vocab_size + v];
                if g_vt != 0.0 {
                    let h_t = &hidden[t];
                    // Safety: hidden[t] must match hidden_size
                    if h_t.len() != hidden_size {
                        return Err(ModelError::Training(format!(
                            "Hidden size {} mismatch at t={} (expected {})",
                            h_t.len(),
                            t,
                            hidden_size
                        )));
                    }
                    for k in 0..hidden_size {
                        row_grad[k] += g_vt * h_t[k];
                    }
                }
            }
            // Apply update to lm_head.weight[v]
            let name = format!("lm_head.weight[{}]", v);
            let row_slice = self.model.lm_head_row_mut(v)?;
            self.optimizer.step_parameter(&name, row_slice, &row_grad)?;
        }

        // 3) Embedding gradients: for each position t, dembed[token_t] += W^T * dlogits_t
        // We compute per-t grad_hidden = W^T * dlogits_t, then update the corresponding embedding row.
        for t in 0..num_samples {
            let token_id = input_ids[t] as usize;
            if token_id >= self.vocab_size {
                continue; // skip OOB
            }
            let dlogits_t = &gradients[t * vocab_size..(t + 1) * vocab_size];

            // grad_hidden = W^T * dlogits_t
            let mut grad_hidden = vec![0.0f32; hidden_size];
            for v in 0..vocab_size {
                let g = dlogits_t[v];
                if g != 0.0 {
                    let w_row = &lm_w_snapshot[v];
                    for k in 0..hidden_size {
                        grad_hidden[k] += w_row[k] * g;
                    }
                }
            }

            // Apply update to embedding row for this token
            let name = format!("embeddings.weight[{}]", token_id);
            let row_slice = self.model.embedding_row_mut(token_id)?;
            self.optimizer
                .step_parameter(&name, row_slice, &grad_hidden)?;
        }

        Ok(())
    }

    /// Get current training step
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Evaluate on a batch of examples
    pub fn evaluate(&mut self, examples: &[TrainingExample]) -> Result<TrainingMetrics> {
        if examples.is_empty() {
            return Err(ModelError::Training("Empty evaluation set".to_string()));
        }

        let (inputs_per_example, targets) = self.prepare_last_step_data(examples)?;
        let (predictions, _last_hiddens, _last_input_ids) =
            self.forward_last_step(&inputs_per_example)?;

        let target_floats: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
        let loss = self.loss_fn.compute_loss(&predictions, &target_floats)?;
        let accuracy = self.loss_fn.compute_accuracy(&predictions, &targets);

        Ok(TrainingMetrics::new(loss, accuracy, self.step_count))
    }
}

/// Reward function for evaluating reasoning quality
pub trait RewardFunction {
    fn compute_reward(&self, reasoning_chain: &[String], target: &str, predicted: &str) -> f32;
}

/// Simple reward function based on correctness and reasoning quality
pub struct SimpleRewardFunction;

impl RewardFunction for SimpleRewardFunction {
    fn compute_reward(&self, reasoning_chain: &[String], target: &str, predicted: &str) -> f32 {
        let mut reward = 0.0;

        // Base reward for correct answer
        if predicted.trim().to_lowercase() == target.trim().to_lowercase() {
            reward += 1.0;
        }

        // Bonus for reasoning quality
        let reasoning_bonus = self.evaluate_reasoning_quality(reasoning_chain);
        reward += reasoning_bonus;

        // Penalty for very short or very long reasoning
        let length_penalty = self.evaluate_reasoning_length(reasoning_chain);
        reward += length_penalty;

        reward.max(0.0) // Ensure non-negative reward
    }
}

impl SimpleRewardFunction {
    /// Evaluate the quality of reasoning steps
    fn evaluate_reasoning_quality(&self, reasoning_chain: &[String]) -> f32 {
        if reasoning_chain.is_empty() {
            return -0.5; // Penalty for no reasoning
        }

        let mut quality_score: f32 = 0.0;

        // Reward for step-by-step structure
        if reasoning_chain.len() >= 2 {
            quality_score += 0.2;
        }

        // Reward for mathematical keywords in math problems
        let math_keywords = ["add", "subtract", "multiply", "solve", "equation", "="];
        let reasoning_text = reasoning_chain.join(" ").to_lowercase();

        for keyword in &math_keywords {
            if reasoning_text.contains(keyword) {
                quality_score += 0.1;
            }
        }

        // Reward for logical connectors
        let logical_connectors = ["therefore", "since", "because", "so", "thus"];
        for connector in &logical_connectors {
            if reasoning_text.contains(connector) {
                quality_score += 0.1;
            }
        }

        quality_score.min(0.5) // Cap the bonus
    }

    /// Evaluate reasoning length appropriateness
    fn evaluate_reasoning_length(&self, reasoning_chain: &[String]) -> f32 {
        let length = reasoning_chain.len();

        match length {
            0 => -0.3,     // Too short
            1 => -0.1,     // Still too short
            2..=5 => 0.0,  // Good length
            6..=8 => -0.1, // Getting long
            _ => -0.2,     // Too long
        }
    }
}

/// Policy gradient computation for REINFORCE algorithm
#[derive(Debug, Clone)]
pub struct PolicyGradient {
    pub action_probs: Vec<f32>,
    pub rewards: Vec<f32>,
    pub baseline: f32,
}

impl PolicyGradient {
    /// Create new policy gradient
    pub fn new(action_probs: Vec<f32>, rewards: Vec<f32>) -> Self {
        let baseline = if rewards.is_empty() {
            0.0
        } else {
            rewards.iter().sum::<f32>() / rewards.len() as f32
        };

        Self {
            action_probs,
            rewards,
            baseline,
        }
    }

    /// Compute policy gradient using REINFORCE
    pub fn compute_gradients(&self) -> Vec<f32> {
        let mut gradients = vec![0.0; self.action_probs.len()];

        for (i, (&prob, &reward)) in self
            .action_probs
            .iter()
            .zip(self.rewards.iter())
            .enumerate()
        {
            if prob > 0.0 {
                // REINFORCE gradient: (reward - baseline) * grad_log_prob
                let advantage = reward - self.baseline;
                gradients[i] = advantage / prob; // Simplified gradient of log probability
            }
        }

        gradients
    }
}

/// Reinforcement learning trainer using REINFORCE algorithm
pub struct RLTrainer {
    model: DeepSeekR1Model,
    optimizer: Optimizer,
    reward_fn: SimpleRewardFunction,
    step_count: usize,
    vocab_size: usize,
    baseline_history: Vec<f32>,
    max_baseline_history: usize,
}

impl RLTrainer {
    /// Create a new RL trainer
    pub fn new(model: DeepSeekR1Model) -> Result<Self> {
        let optimizer_config = OptimizerConfig {
            learning_rate: 1e-5, // Lower learning rate for RL
            ..OptimizerConfig::default()
        };
        let optimizer = Optimizer::new(optimizer_config)?;
        let reward_fn = SimpleRewardFunction;
        let vocab_size = model.config().vocab_size;

        Ok(Self {
            model,
            optimizer,
            reward_fn,
            step_count: 0,
            vocab_size,
            baseline_history: Vec::new(),
            max_baseline_history: 100,
        })
    }

    /// Create RL trainer with custom optimizer config
    pub fn with_optimizer_config(
        model: DeepSeekR1Model,
        optimizer_config: OptimizerConfig,
    ) -> Result<Self> {
        let optimizer = Optimizer::new(optimizer_config)?;
        let reward_fn = SimpleRewardFunction;
        let vocab_size = model.config().vocab_size;

        Ok(Self {
            model,
            optimizer,
            reward_fn,
            step_count: 0,
            vocab_size,
            baseline_history: Vec::new(),
            max_baseline_history: 100,
        })
    }

    /// Generate response with reasoning for RL training
    fn generate_response_with_reasoning(&mut self, input: &str) -> Result<(String, Vec<String>)> {
        // Tokenize input using RLTrainer's tokenizer
        let input_tokens = self.tokenize(input);
        // Forward pass through the model
        let logits = self.model.forward(&input_tokens)?;
        // Decode response using argmax for simplicity
        let response_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0) as u32;
        let response = self.decode(&[response_token]);
        // Reasoning chain extraction (if available, otherwise empty)
        let reasoning_chain = Vec::new(); // TODO: Extract reasoning chain from model output if supported
        Ok((response, reasoning_chain))
    }

    /// Compute action probabilities (simplified)
    fn compute_action_probabilities(&mut self, input: &str) -> Result<Vec<f32>> {
        // Use model logits for probability computation
        let input_tokens = self.tokenize(input);
        let logits = self.model.forward(&input_tokens)?;
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();
        Ok(probs)
    }

    /// Update baseline estimate
    fn update_baseline(&mut self, reward: f32) {
        self.baseline_history.push(reward);

        // Keep only recent history
        if self.baseline_history.len() > self.max_baseline_history {
            self.baseline_history.remove(0);
        }
    }

    /// Get current baseline estimate
    fn get_baseline(&self) -> f32 {
        if self.baseline_history.is_empty() {
            0.0
        } else {
            self.baseline_history.iter().sum::<f32>() / self.baseline_history.len() as f32
        }
    }

    /// Perform an RL training step using REINFORCE
    pub fn train_step(&mut self, batch: &TrainingBatch) -> Result<TrainingMetrics> {
        if batch.examples.is_empty() {
            return Err(ModelError::Training("Empty batch".to_string()));
        }

        let mut total_reward = 0.0;
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;

        for example in &batch.examples {
            // Generate response with reasoning
            let (predicted_response, reasoning_chain) =
                self.generate_response_with_reasoning(&example.input)?;

            // Compute reward
            let reward = self.reward_fn.compute_reward(
                &reasoning_chain,
                &example.target,
                &predicted_response,
            );

            total_reward += reward;
            self.update_baseline(reward);

            // Check if prediction is correct
            if predicted_response.trim().to_lowercase() == example.target.trim().to_lowercase() {
                correct_predictions += 1;
            }

            // Compute action probabilities
            let action_probs = self.compute_action_probabilities(&example.input)?;

            // Compute policy gradient
            let rewards = vec![reward; action_probs.len()];
            let policy_grad = PolicyGradient::new(action_probs, rewards);
            let gradients = policy_grad.compute_gradients();

            // Compute loss (negative expected reward)
            let loss = -(reward - self.get_baseline());
            total_loss += loss;

            // Update parameters using policy gradients
            // In a real implementation, this would update actual model parameters
            let mut dummy_params = vec![0.1; gradients.len()];
            self.optimizer.step_parameter(
                &format!("rl_params_{}", example.input.len()),
                &mut dummy_params,
                &gradients,
            )?;
        }

        self.step_count += 1;

        let _avg_reward = total_reward / batch.examples.len() as f32;
        let avg_loss = total_loss / batch.examples.len() as f32;
        let accuracy = correct_predictions as f32 / batch.examples.len() as f32;

        Ok(TrainingMetrics::new(avg_loss, accuracy, self.step_count))
    }

    /// Evaluate the RL policy on examples
    pub fn evaluate(&mut self, examples: &[TrainingExample]) -> Result<RLEvaluationMetrics> {
        if examples.is_empty() {
            return Err(ModelError::Training("Empty evaluation set".to_string()));
        }

        let mut total_reward = 0.0;
        let mut correct_predictions = 0;
        let mut reasoning_quality_scores = Vec::new();

        for example in examples {
            let (predicted_response, reasoning_chain) =
                self.generate_response_with_reasoning(&example.input)?;

            let reward = self.reward_fn.compute_reward(
                &reasoning_chain,
                &example.target,
                &predicted_response,
            );

            total_reward += reward;
            reasoning_quality_scores.push(reward);

            if predicted_response.trim().to_lowercase() == example.target.trim().to_lowercase() {
                correct_predictions += 1;
            }
        }

        let avg_reward = total_reward / examples.len() as f32;
        let accuracy = correct_predictions as f32 / examples.len() as f32;
        let avg_reasoning_quality =
            reasoning_quality_scores.iter().sum::<f32>() / reasoning_quality_scores.len() as f32;

        Ok(RLEvaluationMetrics {
            average_reward: avg_reward,
            accuracy,
            reasoning_quality: avg_reasoning_quality,
            baseline: self.get_baseline(),
            total_examples: examples.len(),
        })
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get current baseline
    pub fn baseline(&self) -> f32 {
        self.get_baseline()
    }
}

impl RLTrainer {
    /// Tokenize text using word-based hashing (same as BasicTrainer)
    fn tokenize(&self, text: &str) -> Vec<u32> {
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding.split_whitespace().collect();

        let mut token_ids = Vec::new();

        for word in words {
            let mut hash = 0u32;
            for byte in word.bytes() {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
            }
            token_ids.push(hash % self.vocab_size as u32);
        }

        if token_ids.is_empty() {
            token_ids.push(0); // UNK token
        }

        token_ids
    }

    /// Decode token IDs to a string (simple implementation)
    fn decode(&self, token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .map(|id| format!("<{}>", id))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// RL-specific evaluation metrics
#[derive(Debug, Clone)]
pub struct RLEvaluationMetrics {
    pub average_reward: f32,
    pub accuracy: f32,
    pub reasoning_quality: f32,
    pub baseline: f32,
    pub total_examples: usize,
}

impl RLEvaluationMetrics {
    /// Display metrics in a formatted way
    pub fn display(&self) {
        println!("RL Evaluation Metrics:");
        println!("  Average Reward: {:.4}", self.average_reward);
        println!("  Accuracy: {:.2}%", self.accuracy * 100.0);
        println!("  Reasoning Quality: {:.4}", self.reasoning_quality);
        println!("  Baseline: {:.4}", self.baseline);
        println!("  Total Examples: {}", self.total_examples);
        /// Tokenize text using word-based hashing (same as BasicTrainer)
        fn tokenize(text: &str, vocab_size: usize) -> Vec<u32> {
            let binding = text.to_lowercase();
            let words: Vec<&str> = binding.split_whitespace().collect();

            let mut token_ids = Vec::new();

            for word in words {
                let mut hash = 0u32;
                for byte in word.bytes() {
                    hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
                }
                token_ids.push(hash % vocab_size as u32);
            }

            if token_ids.is_empty() {
                token_ids.push(0); // UNK token
            }

            token_ids
        }

        /// Decode token IDs to a string (simple implementation)
        fn decode(token_ids: &[u32]) -> String {
            token_ids
                .iter()
                .map(|id| format!("<{}>", id))
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeepSeekR1Model, ModelConfig};
    use crate::training::data::{ProblemType, TrainingExample};

    #[test]
    fn test_basic_trainer_creation() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let trainer = BasicTrainer::new(model);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_basic_trainer_with_custom_config() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();

        let optimizer_config = OptimizerConfig {
            learning_rate: 0.001,
            ..OptimizerConfig::default()
        };

        let trainer = BasicTrainer::with_optimizer_config(model, optimizer_config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_training_step() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut trainer = BasicTrainer::new(model).unwrap();

        let examples = vec![
            TrainingExample::new("2 + 2".to_string(), "4".to_string(), ProblemType::Math),
            TrainingExample::new("3 * 3".to_string(), "9".to_string(), ProblemType::Math),
        ];

        let batch = TrainingBatch::new(examples);
        let result = trainer.train_step(&batch);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.loss >= 0.0);
        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        assert_eq!(metrics.step, 1);
    }

    #[test]
    fn test_evaluation() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut trainer = BasicTrainer::new(model).unwrap();

        let examples = vec![TrainingExample::new(
            "test".to_string(),
            "result".to_string(),
            ProblemType::General,
        )];

        let result = trainer.evaluate(&examples);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_batch_error() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut trainer = BasicTrainer::new(model).unwrap();

        let batch = TrainingBatch::new(vec![]);
        let result = trainer.train_step(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_rl_trainer_creation() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let trainer = RLTrainer::new(model);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_rl_trainer_with_custom_config() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();

        let optimizer_config = OptimizerConfig {
            learning_rate: 1e-6,
            ..OptimizerConfig::default()
        };

        let trainer = RLTrainer::with_optimizer_config(model, optimizer_config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_reward_function() {
        let reward_fn = SimpleRewardFunction;

        // Test correct answer with good reasoning
        let reasoning = vec![
            "I need to add 2 and 2".to_string(),
            "2 + 2 = 4".to_string(),
            "Therefore, the answer is 4".to_string(),
        ];
        let reward = reward_fn.compute_reward(&reasoning, "4", "4");
        assert!(reward > 1.0); // Should get base reward + bonuses

        // Test incorrect answer
        let reward_wrong = reward_fn.compute_reward(&reasoning, "4", "5");
        assert!(reward_wrong < reward); // Should be lower than correct answer

        // Test no reasoning
        let reward_no_reasoning = reward_fn.compute_reward(&[], "4", "4");
        assert!(reward_no_reasoning < reward); // Should be penalized for no reasoning
    }

    #[test]
    fn test_policy_gradient() {
        let action_probs = vec![0.3, 0.5, 0.2];
        let rewards = vec![1.0, 0.5, 0.8];

        let policy_grad = PolicyGradient::new(action_probs, rewards);
        assert!((policy_grad.baseline - 0.767).abs() < 0.01); // Average of rewards

        let gradients = policy_grad.compute_gradients();
        assert_eq!(gradients.len(), 3);
    }

    #[test]
    fn test_rl_training_step() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut trainer = RLTrainer::new(model).unwrap();

        let examples = vec![TrainingExample::new(
            "2 + 2".to_string(),
            "4".to_string(),
            ProblemType::Math,
        )];

        let batch = TrainingBatch::new(examples);
        let result = trainer.train_step(&batch);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_eq!(metrics.step, 1);
    }

    #[test]
    fn test_rl_evaluation() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut trainer = RLTrainer::new(model).unwrap();

        let examples = vec![TrainingExample::new(
            "test".to_string(),
            "result".to_string(),
            ProblemType::General,
        )];

        let result = trainer.evaluate(&examples);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_eq!(metrics.total_examples, 1);
        assert!(metrics.average_reward >= 0.0);
    }

    #[test]
    fn test_loss_decreases_after_one_step() {
        let config = ModelConfig::default();
        let model = DeepSeekR1Model::new(config).unwrap();
        let mut trainer = BasicTrainer::new(model).unwrap();

        // Small deterministic batch
        let examples = vec![
            TrainingExample::new(
                "What is 2 + 2?".to_string(),
                "4".to_string(),
                ProblemType::Math,
            ),
            TrainingExample::new(
                "Spell 'cat'".to_string(),
                "cat".to_string(),
                ProblemType::General,
            ),
            TrainingExample::new("3 * 3 = ?".to_string(), "9".to_string(), ProblemType::Math),
            TrainingExample::new(
                "Hello".to_string(),
                "Hello".to_string(),
                ProblemType::General,
            ),
        ];

        // Baseline loss before training
        let baseline = trainer.evaluate(&examples).unwrap().loss;

        // One training step on the same batch
        let batch = TrainingBatch::new(examples.clone());
        let _metrics = trainer.train_step(&batch).unwrap();

        // Loss after one update should decrease (tolerance-based)
        let after = trainer.evaluate(&examples).unwrap().loss;
        assert!(
            after < baseline,
            "expected loss to decrease: before {:.6} after {:.6}",
            baseline,
            after
        );
    }
}
