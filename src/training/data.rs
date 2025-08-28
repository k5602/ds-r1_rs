//! # Training Data
//!
//! Data structures and utilities for training data management.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Type of problem for training
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProblemType {
    Math,
    Code,
    Logic,
    General,
}

/// Individual training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub input: String,
    pub target: String,
    pub reasoning_chain: Option<Vec<String>>,
    pub problem_type: ProblemType,
}

impl TrainingExample {
    /// Create a new training example
    pub fn new(input: String, target: String, problem_type: ProblemType) -> Self {
        Self {
            input,
            target,
            reasoning_chain: None,
            problem_type,
        }
    }

    /// Create a training example with reasoning chain
    pub fn with_reasoning(
        input: String,
        target: String,
        reasoning_chain: Vec<String>,
        problem_type: ProblemType,
    ) -> Self {
        Self {
            input,
            target,
            reasoning_chain: Some(reasoning_chain),
            problem_type,
        }
    }
}

/// Batch of training examples
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub examples: Vec<TrainingExample>,
    pub batch_size: usize,
}

impl TrainingBatch {
    /// Create a new training batch
    pub fn new(examples: Vec<TrainingExample>) -> Self {
        let batch_size = examples.len();
        Self {
            examples,
            batch_size,
        }
    }

    /// Split examples by problem type
    pub fn split_by_type(&self) -> std::collections::HashMap<ProblemType, Vec<&TrainingExample>> {
        let mut map = std::collections::HashMap::new();

        for example in &self.examples {
            map.entry(example.problem_type.clone())
                .or_insert_with(Vec::new)
                .push(example);
        }

        map
    }
}

/// Synthetic dataset generator for training
pub struct SyntheticDataGenerator {
    rng: rand::rngs::ThreadRng,
}

impl SyntheticDataGenerator {
    /// Create a new synthetic data generator
    pub fn new() -> Self {
        Self { rng: rand::rng() }
    }

    /// Generate mathematical reasoning problems
    pub fn generate_math_problems(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let problem_type = self.rng.random_range(0..4);

            let example = match problem_type {
                0 => self.generate_addition_problem(),
                1 => self.generate_subtraction_problem(),
                2 => self.generate_multiplication_problem(),
                3 => self.generate_simple_equation(),
                _ => unreachable!(),
            };

            examples.push(example);
        }

        examples
    }

    /// Generate addition problems
    fn generate_addition_problem(&mut self) -> TrainingExample {
        let a = self.rng.random_range(1..100);
        let b = self.rng.random_range(1..100);
        let result = a + b;

        let input = format!("What is {} + {}?", a, b);
        let reasoning = vec![
            format!("I need to add {} and {}", a, b),
            format!("{} + {} = {}", a, b, result),
        ];
        let target = format!("{}", result);

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Math)
    }

    /// Generate subtraction problems
    fn generate_subtraction_problem(&mut self) -> TrainingExample {
        let a = self.rng.random_range(50..200);
        let b = self.rng.random_range(1..a);
        let result = a - b;

        let input = format!("What is {} - {}?", a, b);
        let reasoning = vec![
            format!("I need to subtract {} from {}", b, a),
            format!("{} - {} = {}", a, b, result),
        ];
        let target = format!("{}", result);

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Math)
    }

    /// Generate multiplication problems
    fn generate_multiplication_problem(&mut self) -> TrainingExample {
        let a = self.rng.random_range(2..20);
        let b = self.rng.random_range(2..20);
        let result = a * b;

        let input = format!("What is {} × {}?", a, b);
        let reasoning = vec![
            format!("I need to multiply {} by {}", a, b),
            format!("{} × {} = {}", a, b, result),
        ];
        let target = format!("{}", result);

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Math)
    }

    /// Generate simple linear equations
    fn generate_simple_equation(&mut self) -> TrainingExample {
        let x = self.rng.random_range(1..20);
        let b = self.rng.random_range(1..50);
        let result = x + b;

        let input = format!("Solve for x: x + {} = {}", b, result);
        let reasoning = vec![
            format!("I need to solve x + {} = {}", b, result),
            format!("Subtracting {} from both sides: x = {} - {}", b, result, b),
            format!("Therefore: x = {}", x),
        ];
        let target = format!("x = {}", x);

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Math)
    }

    /// Generate code explanation examples
    pub fn generate_code_examples(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let example_type = self.rng.random_range(0..3);

            let example = match example_type {
                0 => self.generate_loop_explanation(),
                1 => self.generate_function_explanation(),
                2 => self.generate_conditional_explanation(),
                _ => unreachable!(),
            };

            examples.push(example);
        }

        examples
    }

    /// Generate loop explanation
    fn generate_loop_explanation(&mut self) -> TrainingExample {
        let n = self.rng.random_range(3..10);
        let code = format!("for i in range({}):\n    print(i)", n);

        let input = format!("Explain what this code does:\n{}", code);
        let reasoning = vec![
            "This is a for loop in Python".to_string(),
            format!("It iterates from 0 to {} (exclusive)", n),
            "In each iteration, it prints the current value of i".to_string(),
            format!("So it will print numbers 0, 1, 2, ..., {}", n - 1),
        ];
        let target = format!(
            "This code prints numbers from 0 to {} using a for loop",
            n - 1
        );

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Code)
    }

    /// Generate function explanation
    fn generate_function_explanation(&mut self) -> TrainingExample {
        let code = "def add_numbers(a, b):\n    return a + b";

        let input = format!("Explain what this function does:\n{}", code);
        let reasoning = vec![
            "This defines a function called 'add_numbers'".to_string(),
            "It takes two parameters: 'a' and 'b'".to_string(),
            "The function returns the sum of a and b".to_string(),
            "This is a simple addition function".to_string(),
        ];
        let target = "This function takes two numbers and returns their sum".to_string();

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Code)
    }

    /// Generate conditional explanation
    fn generate_conditional_explanation(&mut self) -> TrainingExample {
        let threshold = self.rng.random_range(10..100);
        let code = format!(
            "if x > {}:\n    print('Large')\nelse:\n    print('Small')",
            threshold
        );

        let input = format!("Explain what this code does:\n{}", code);
        let reasoning = vec![
            "This is an if-else conditional statement".to_string(),
            format!("It checks if variable x is greater than {}", threshold),
            "If x is greater, it prints 'Large'".to_string(),
            "Otherwise, it prints 'Small'".to_string(),
        ];
        let target = format!(
            "This code prints 'Large' if x > {}, otherwise prints 'Small'",
            threshold
        );

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Code)
    }

    /// Generate logical reasoning tasks
    pub fn generate_logic_problems(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let problem_type = self.rng.random_range(0..3);

            let example = match problem_type {
                0 => self.generate_syllogism(),
                1 => self.generate_pattern_recognition(),
                2 => self.generate_simple_deduction(),
                _ => unreachable!(),
            };

            examples.push(example);
        }

        examples
    }

    /// Generate syllogism problems
    fn generate_syllogism(&mut self) -> TrainingExample {
        let animals = ["cats", "dogs", "birds", "fish"];
        let properties = ["mammals", "vertebrates", "animals", "living things"];

        let animal = animals[self.rng.random_range(0..animals.len())];
        let property = properties[self.rng.random_range(0..properties.len())];

        let input = format!(
            "All {} are {}. Fluffy is a cat. Is Fluffy a {}?",
            animal, property, property
        );
        let reasoning = vec![
            format!("Given: All {} are {}", animal, property),
            "Given: Fluffy is a cat".to_string(),
            format!("Since cats are {}, and Fluffy is a cat", property),
            format!("Therefore: Fluffy is a {}", property),
        ];
        let target = format!("Yes, Fluffy is a {}", property);

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Logic)
    }

    /// Generate pattern recognition
    fn generate_pattern_recognition(&mut self) -> TrainingExample {
        let start = self.rng.random_range(1..10);
        let step = self.rng.random_range(2..5);
        let sequence: Vec<i32> = (0..4).map(|i| start + i * step).collect();
        let next = start + 4 * step;

        let input = format!(
            "What comes next in this sequence: {}, {}, {}, {}?",
            sequence[0], sequence[1], sequence[2], sequence[3]
        );
        let reasoning = vec![
            format!(
                "Looking at the sequence: {}, {}, {}, {}",
                sequence[0], sequence[1], sequence[2], sequence[3]
            ),
            format!("The difference between consecutive terms is {}", step),
            format!("This is an arithmetic sequence with step {}", step),
            format!(
                "The next term would be {} + {} = {}",
                sequence[3], step, next
            ),
        ];
        let target = format!("{}", next);

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Logic)
    }

    /// Generate simple deduction
    fn generate_simple_deduction(&mut self) -> TrainingExample {
        let colors = ["red", "blue", "green", "yellow"];
        let objects = ["car", "house", "ball", "book"];

        let color = colors[self.rng.random_range(0..colors.len())];
        let object = objects[self.rng.random_range(0..objects.len())];

        let input = format!(
            "If all {} things are expensive, and this {} is {}, is it expensive?",
            color, object, color
        );
        let reasoning = vec![
            format!("Given: All {} things are expensive", color),
            format!("Given: This {} is {}", object, color),
            format!(
                "Since the {} is {}, and all {} things are expensive",
                object, color, color
            ),
            format!("Therefore: This {} is expensive", object),
        ];
        let target = "Yes, it is expensive".to_string();

        TrainingExample::with_reasoning(input, target, reasoning, ProblemType::Logic)
    }

    /// Generate a mixed dataset with all problem types
    pub fn generate_mixed_dataset(&mut self, total_count: usize) -> Vec<TrainingExample> {
        let math_count = total_count / 3;
        let code_count = total_count / 3;
        let logic_count = total_count - math_count - code_count;

        let mut examples = Vec::new();
        examples.extend(self.generate_math_problems(math_count));
        examples.extend(self.generate_code_examples(code_count));
        examples.extend(self.generate_logic_problems(logic_count));

        // Shuffle the examples
        use rand::seq::SliceRandom;
        examples.shuffle(&mut self.rng);

        examples
    }
}

impl Default for SyntheticDataGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Data loader for batching training examples
pub struct DataLoader {
    examples: Vec<TrainingExample>,
    batch_size: usize,
    current_index: usize,
    shuffle: bool,
    rng: rand::rngs::ThreadRng,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new(examples: Vec<TrainingExample>, batch_size: usize, shuffle: bool) -> Self {
        Self {
            examples,
            batch_size,
            current_index: 0,
            shuffle,
            rng: rand::rng(),
        }
    }

    /// Reset the data loader to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            self.examples.shuffle(&mut self.rng);
        }
    }

    /// Get the next batch
    pub fn next_batch(&mut self) -> Option<TrainingBatch> {
        if self.current_index >= self.examples.len() {
            return None;
        }

        let end_index = (self.current_index + self.batch_size).min(self.examples.len());
        let batch_examples = self.examples[self.current_index..end_index].to_vec();
        self.current_index = end_index;

        Some(TrainingBatch::new(batch_examples))
    }

    /// Check if there are more batches
    pub fn has_next(&self) -> bool {
        self.current_index < self.examples.len()
    }

    /// Get total number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if the data loader is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get number of batches per epoch
    pub fn batches_per_epoch(&self) -> usize {
        (self.examples.len() + self.batch_size - 1) / self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_example_creation() {
        let example =
            TrainingExample::new("2 + 2 = ?".to_string(), "4".to_string(), ProblemType::Math);

        assert_eq!(example.input, "2 + 2 = ?");
        assert_eq!(example.target, "4");
        assert!(matches!(example.problem_type, ProblemType::Math));
        assert!(example.reasoning_chain.is_none());
    }

    #[test]
    fn test_training_example_with_reasoning() {
        let reasoning = vec!["I need to add 2 and 2".to_string(), "2 + 2 = 4".to_string()];

        let example = TrainingExample::with_reasoning(
            "2 + 2 = ?".to_string(),
            "4".to_string(),
            reasoning.clone(),
            ProblemType::Math,
        );

        assert_eq!(example.reasoning_chain, Some(reasoning));
    }

    #[test]
    fn test_training_batch() {
        let examples = vec![
            TrainingExample::new("2 + 2".to_string(), "4".to_string(), ProblemType::Math),
            TrainingExample::new("3 * 3".to_string(), "9".to_string(), ProblemType::Math),
        ];

        let batch = TrainingBatch::new(examples);
        assert_eq!(batch.batch_size, 2);

        let by_type = batch.split_by_type();
        assert_eq!(by_type.len(), 1);
        assert!(by_type.contains_key(&ProblemType::Math));
    }

    #[test]
    fn test_synthetic_data_generator() {
        let mut generator = SyntheticDataGenerator::new();

        // Test math problems generation
        let math_problems = generator.generate_math_problems(5);
        assert_eq!(math_problems.len(), 5);
        for problem in &math_problems {
            assert!(matches!(problem.problem_type, ProblemType::Math));
            assert!(problem.reasoning_chain.is_some());
        }

        // Test code examples generation
        let code_examples = generator.generate_code_examples(3);
        assert_eq!(code_examples.len(), 3);
        for example in &code_examples {
            assert!(matches!(example.problem_type, ProblemType::Code));
            assert!(example.reasoning_chain.is_some());
        }

        // Test logic problems generation
        let logic_problems = generator.generate_logic_problems(4);
        assert_eq!(logic_problems.len(), 4);
        for problem in &logic_problems {
            assert!(matches!(problem.problem_type, ProblemType::Logic));
            assert!(problem.reasoning_chain.is_some());
        }
    }

    #[test]
    fn test_mixed_dataset_generation() {
        let mut generator = SyntheticDataGenerator::new();
        let mixed_dataset = generator.generate_mixed_dataset(12);

        assert_eq!(mixed_dataset.len(), 12);

        // Check that we have different problem types
        let mut has_math = false;
        let mut has_code = false;
        let mut has_logic = false;

        for example in &mixed_dataset {
            match example.problem_type {
                ProblemType::Math => has_math = true,
                ProblemType::Code => has_code = true,
                ProblemType::Logic => has_logic = true,
                _ => {}
            }
        }

        assert!(has_math);
        assert!(has_code);
        assert!(has_logic);
    }

    #[test]
    fn test_data_loader() {
        let examples = vec![
            TrainingExample::new("1".to_string(), "a".to_string(), ProblemType::Math),
            TrainingExample::new("2".to_string(), "b".to_string(), ProblemType::Math),
            TrainingExample::new("3".to_string(), "c".to_string(), ProblemType::Math),
            TrainingExample::new("4".to_string(), "d".to_string(), ProblemType::Math),
            TrainingExample::new("5".to_string(), "e".to_string(), ProblemType::Math),
        ];

        let mut loader = DataLoader::new(examples, 2, false);

        assert_eq!(loader.len(), 5);
        assert_eq!(loader.batches_per_epoch(), 3);
        assert!(loader.has_next());

        // Get first batch
        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1.batch_size, 2);

        // Get second batch
        let batch2 = loader.next_batch().unwrap();
        assert_eq!(batch2.batch_size, 2);

        // Get third batch (partial)
        let batch3 = loader.next_batch().unwrap();
        assert_eq!(batch3.batch_size, 1);

        // No more batches
        assert!(!loader.has_next());
        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_data_loader_reset() {
        let examples = vec![
            TrainingExample::new("1".to_string(), "a".to_string(), ProblemType::Math),
            TrainingExample::new("2".to_string(), "b".to_string(), ProblemType::Math),
        ];

        let mut loader = DataLoader::new(examples, 1, false);

        // Consume all batches
        let _batch1 = loader.next_batch().unwrap();
        let _batch2 = loader.next_batch().unwrap();
        assert!(!loader.has_next());

        // Reset and check we can iterate again
        loader.reset();
        assert!(loader.has_next());
        let _batch1_again = loader.next_batch().unwrap();
        assert!(loader.has_next());
    }

    #[test]
    fn test_empty_data_loader() {
        let loader = DataLoader::new(vec![], 5, false);
        assert!(loader.is_empty());
        assert_eq!(loader.len(), 0);
        assert_eq!(loader.batches_per_epoch(), 0);
    }
}
