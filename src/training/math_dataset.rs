//! # Mathematical Problem Dataset
//!
//! Collection of mathematical problems for training and evaluation.

use crate::inference::math_solver::MathProblemType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A mathematical problem with expected solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathProblem {
    pub id: String,
    pub problem_text: String,
    pub problem_type: MathProblemType,
    pub difficulty: DifficultyLevel,
    pub expected_answer: String,
    pub expected_steps: Vec<String>,
    pub reasoning_chain: Option<String>,
    pub tags: Vec<String>,
}

/// Difficulty levels for mathematical problems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Elementary,
    Intermediate,
    Advanced,
}

/// Mathematical problem dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathDataset {
    pub problems: Vec<MathProblem>,
    pub metadata: DatasetMetadata,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub total_problems: usize,
    pub problem_type_counts: HashMap<String, usize>,
    pub difficulty_counts: HashMap<String, usize>,
}

impl MathProblem {
    /// Create a new mathematical problem
    pub fn new(
        id: String,
        problem_text: String,
        problem_type: MathProblemType,
        difficulty: DifficultyLevel,
        expected_answer: String,
    ) -> Self {
        Self {
            id,
            problem_text,
            problem_type,
            difficulty,
            expected_answer,
            expected_steps: Vec::new(),
            reasoning_chain: None,
            tags: Vec::new(),
        }
    }

    /// Add expected reasoning steps
    pub fn with_steps(mut self, steps: Vec<String>) -> Self {
        self.expected_steps = steps;
        self
    }

    /// Add reasoning chain
    pub fn with_reasoning_chain(mut self, reasoning: String) -> Self {
        self.reasoning_chain = Some(reasoning);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if the problem matches given criteria
    pub fn matches_criteria(&self, problem_type: Option<&MathProblemType>, difficulty: Option<&DifficultyLevel>) -> bool {
        let type_match = problem_type.map_or(true, |t| &self.problem_type == t);
        let difficulty_match = difficulty.map_or(true, |d| &self.difficulty == d);
        type_match && difficulty_match
    }
}

impl MathDataset {
    /// Create a new empty dataset
    pub fn new(name: String, version: String, description: String) -> Self {
        Self {
            problems: Vec::new(),
            metadata: DatasetMetadata {
                name,
                version,
                description,
                total_problems: 0,
                problem_type_counts: HashMap::new(),
                difficulty_counts: HashMap::new(),
            },
        }
    }

    /// Add a problem to the dataset
    pub fn add_problem(&mut self, problem: MathProblem) {
        // Update counts
        let type_key = format!("{:?}", problem.problem_type);
        let difficulty_key = format!("{:?}", problem.difficulty);
        
        *self.metadata.problem_type_counts.entry(type_key).or_insert(0) += 1;
        *self.metadata.difficulty_counts.entry(difficulty_key).or_insert(0) += 1;
        
        self.problems.push(problem);
        self.metadata.total_problems = self.problems.len();
    }

    /// Get problems by type
    pub fn get_problems_by_type(&self, problem_type: &MathProblemType) -> Vec<&MathProblem> {
        self.problems
            .iter()
            .filter(|p| &p.problem_type == problem_type)
            .collect()
    }

    /// Get problems by difficulty
    pub fn get_problems_by_difficulty(&self, difficulty: &DifficultyLevel) -> Vec<&MathProblem> {
        self.problems
            .iter()
            .filter(|p| &p.difficulty == difficulty)
            .collect()
    }

    /// Get problems matching criteria
    pub fn get_problems_matching(&self, problem_type: Option<&MathProblemType>, difficulty: Option<&DifficultyLevel>) -> Vec<&MathProblem> {
        self.problems
            .iter()
            .filter(|p| p.matches_criteria(problem_type, difficulty))
            .collect()
    }

    /// Get a random sample of problems
    pub fn get_random_sample(&self, count: usize) -> Vec<&MathProblem> {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        let mut problems: Vec<&MathProblem> = self.problems.iter().collect();
        problems.shuffle(&mut rng);
        problems.into_iter().take(count).collect()
    }

    /// Create the default mathematical dataset
    pub fn create_default_dataset() -> Self {
        let mut dataset = Self::new(
            "DeepSeek R1 Math Dataset".to_string(),
            "1.0.0".to_string(),
            "Comprehensive mathematical problem dataset for reasoning training".to_string(),
        );

        // Add arithmetic problems
        dataset.add_arithmetic_problems();
        
        // Add algebraic problems
        dataset.add_algebraic_problems();
        
        // Add word problems
        dataset.add_word_problems();
        
        // Add equation problems
        dataset.add_equation_problems();

        dataset
    }

    /// Add basic arithmetic problems
    fn add_arithmetic_problems(&mut self) {
        let arithmetic_problems = vec![
            // Elementary addition
            ("arith_001", "What is 15 + 27?", "42", vec![
                "Identify the numbers: 15 and 27".to_string(),
                "Add the numbers: 15 + 27 = 42".to_string(),
            ]),
            ("arith_002", "Calculate 84 - 39", "45", vec![
                "Identify the numbers: 84 and 39".to_string(),
                "Subtract: 84 - 39 = 45".to_string(),
            ]),
            ("arith_003", "What is 12 times 8?", "96", vec![
                "Identify the numbers: 12 and 8".to_string(),
                "Multiply: 12 × 8 = 96".to_string(),
            ]),
            ("arith_004", "Divide 144 by 12", "12", vec![
                "Identify the numbers: 144 and 12".to_string(),
                "Divide: 144 ÷ 12 = 12".to_string(),
            ]),
            
            // Intermediate arithmetic
            ("arith_005", "What is 156 + 289 + 73?", "518", vec![
                "Add the first two numbers: 156 + 289 = 445".to_string(),
                "Add the third number: 445 + 73 = 518".to_string(),
            ]),
            ("arith_006", "Calculate 1000 - 347", "653", vec![
                "Subtract: 1000 - 347 = 653".to_string(),
            ]),
            ("arith_007", "What is 25 × 16?", "400", vec![
                "Multiply: 25 × 16 = 400".to_string(),
            ]),
            ("arith_008", "Divide 2048 by 32", "64", vec![
                "Divide: 2048 ÷ 32 = 64".to_string(),
            ]),
        ];

        for (id, problem, answer, steps) in arithmetic_problems {
            let difficulty = if id.ends_with("001") || id.ends_with("002") || id.ends_with("003") || id.ends_with("004") {
                DifficultyLevel::Elementary
            } else {
                DifficultyLevel::Intermediate
            };

            let math_problem = MathProblem::new(
                id.to_string(),
                problem.to_string(),
                MathProblemType::Arithmetic,
                difficulty,
                answer.to_string(),
            )
            .with_steps(steps)
            .with_tags(vec!["arithmetic".to_string(), "basic_math".to_string()]);

            self.add_problem(math_problem);
        }
    }

    /// Add algebraic problems
    fn add_algebraic_problems(&mut self) {
        let algebra_problems = vec![
            // Elementary algebra
            ("alg_001", "Find the value of x when x + 5 = 12", "x = 7", vec![
                "Identify the equation: x + 5 = 12".to_string(),
                "Subtract 5 from both sides: x = 12 - 5".to_string(),
                "Calculate: x = 7".to_string(),
            ]),
            ("alg_002", "Solve for x: 3x = 21", "x = 7", vec![
                "Identify the equation: 3x = 21".to_string(),
                "Divide both sides by 3: x = 21 ÷ 3".to_string(),
                "Calculate: x = 7".to_string(),
            ]),
            ("alg_003", "What is x if x - 8 = 15?", "x = 23", vec![
                "Identify the equation: x - 8 = 15".to_string(),
                "Add 8 to both sides: x = 15 + 8".to_string(),
                "Calculate: x = 23".to_string(),
            ]),
            
            // Intermediate algebra
            ("alg_004", "Solve: 2x + 7 = 19", "x = 6", vec![
                "Identify the equation: 2x + 7 = 19".to_string(),
                "Subtract 7 from both sides: 2x = 19 - 7 = 12".to_string(),
                "Divide both sides by 2: x = 12 ÷ 2 = 6".to_string(),
            ]),
            ("alg_005", "Find x: 5x - 3 = 22", "x = 5", vec![
                "Identify the equation: 5x - 3 = 22".to_string(),
                "Add 3 to both sides: 5x = 22 + 3 = 25".to_string(),
                "Divide both sides by 5: x = 25 ÷ 5 = 5".to_string(),
            ]),
            ("alg_006", "Solve: 4x + 8 = 3x + 15", "x = 7", vec![
                "Identify the equation: 4x + 8 = 3x + 15".to_string(),
                "Subtract 3x from both sides: x + 8 = 15".to_string(),
                "Subtract 8 from both sides: x = 15 - 8 = 7".to_string(),
            ]),
        ];

        for (id, problem, answer, steps) in algebra_problems {
            let difficulty = if id.ends_with("001") || id.ends_with("002") || id.ends_with("003") {
                DifficultyLevel::Elementary
            } else {
                DifficultyLevel::Intermediate
            };

            let math_problem = MathProblem::new(
                id.to_string(),
                problem.to_string(),
                MathProblemType::Algebra,
                difficulty,
                answer.to_string(),
            )
            .with_steps(steps)
            .with_tags(vec!["algebra".to_string(), "equations".to_string()]);

            self.add_problem(math_problem);
        }
    }

    /// Add word problems
    fn add_word_problems(&mut self) {
        let word_problems = vec![
            // Elementary word problems
            ("word_001", "John has 15 apples and Mary has 23 apples. How many apples do they have in total?", "38", vec![
                "Identify the given information: John has 15 apples, Mary has 23 apples".to_string(),
                "Determine what to find: total number of apples".to_string(),
                "Set up the calculation: 15 + 23".to_string(),
                "Calculate: 15 + 23 = 38".to_string(),
            ]),
            ("word_002", "A store sold 45 items in the morning and 38 items in the afternoon. What is the total number of items sold?", "83", vec![
                "Identify the given information: 45 items in morning, 38 items in afternoon".to_string(),
                "Determine what to find: total items sold".to_string(),
                "Set up the calculation: 45 + 38".to_string(),
                "Calculate: 45 + 38 = 83".to_string(),
            ]),
            ("word_003", "Sarah bought 8 books and each book costs 12 dollars. How much did she spend in total?", "96", vec![
                "Identify the given information: 8 books, each costs $12".to_string(),
                "Determine what to find: total cost".to_string(),
                "Set up the calculation: 8 × 12".to_string(),
                "Calculate: 8 × 12 = 96".to_string(),
            ]),
            ("word_004", "There are 100 students in a school. If 35 are boys, how many are girls?", "65", vec![
                "Identify the given information: 100 total students, 35 boys".to_string(),
                "Determine what to find: number of girls".to_string(),
                "Set up the calculation: 100 - 35".to_string(),
                "Calculate: 100 - 35 = 65".to_string(),
            ]),
            
            // Intermediate word problems
            ("word_005", "A train travels 60 miles per hour for 3.5 hours. How far does it travel?", "210", vec![
                "Identify the given information: speed = 60 mph, time = 3.5 hours".to_string(),
                "Use the formula: distance = speed × time".to_string(),
                "Set up the calculation: 60 × 3.5".to_string(),
                "Calculate: 60 × 3.5 = 210 miles".to_string(),
            ]),
            ("word_006", "A rectangle has a length of 15 meters and a width of 8 meters. What is its area?", "120", vec![
                "Identify the given information: length = 15m, width = 8m".to_string(),
                "Use the formula: area = length × width".to_string(),
                "Set up the calculation: 15 × 8".to_string(),
                "Calculate: 15 × 8 = 120 square meters".to_string(),
            ]),
            ("word_007", "If 5 pencils cost $3.75, how much does one pencil cost?", "0.75", vec![
                "Identify the given information: 5 pencils cost $3.75".to_string(),
                "Determine what to find: cost per pencil".to_string(),
                "Set up the calculation: $3.75 ÷ 5".to_string(),
                "Calculate: $3.75 ÷ 5 = $0.75".to_string(),
            ]),
        ];

        for (id, problem, answer, steps) in word_problems {
            let difficulty = if id.ends_with("001") || id.ends_with("002") || id.ends_with("003") || id.ends_with("004") {
                DifficultyLevel::Elementary
            } else {
                DifficultyLevel::Intermediate
            };

            let math_problem = MathProblem::new(
                id.to_string(),
                problem.to_string(),
                MathProblemType::WordProblem,
                difficulty,
                answer.to_string(),
            )
            .with_steps(steps)
            .with_tags(vec!["word_problem".to_string(), "real_world".to_string()]);

            self.add_problem(math_problem);
        }
    }

    /// Add equation problems
    fn add_equation_problems(&mut self) {
        let equation_problems = vec![
            // Elementary equations
            ("eq_001", "Solve: x + 3 = 7", "x = 4", vec![
                "Identify the equation: x + 3 = 7".to_string(),
                "Subtract 3 from both sides: x = 7 - 3".to_string(),
                "Calculate: x = 4".to_string(),
                "Verify: 4 + 3 = 7 ✓".to_string(),
            ]),
            ("eq_002", "Solve: 2x = 10", "x = 5", vec![
                "Identify the equation: 2x = 10".to_string(),
                "Divide both sides by 2: x = 10 ÷ 2".to_string(),
                "Calculate: x = 5".to_string(),
                "Verify: 2 × 5 = 10 ✓".to_string(),
            ]),
            ("eq_003", "Solve: x - 5 = 12", "x = 17", vec![
                "Identify the equation: x - 5 = 12".to_string(),
                "Add 5 to both sides: x = 12 + 5".to_string(),
                "Calculate: x = 17".to_string(),
                "Verify: 17 - 5 = 12 ✓".to_string(),
            ]),
            
            // Intermediate equations
            ("eq_004", "Solve: 2x + 5 = 13", "x = 4", vec![
                "Identify the equation: 2x + 5 = 13".to_string(),
                "Subtract 5 from both sides: 2x = 13 - 5 = 8".to_string(),
                "Divide both sides by 2: x = 8 ÷ 2 = 4".to_string(),
                "Verify: 2(4) + 5 = 8 + 5 = 13 ✓".to_string(),
            ]),
            ("eq_005", "Solve: 3x - 7 = 14", "x = 7", vec![
                "Identify the equation: 3x - 7 = 14".to_string(),
                "Add 7 to both sides: 3x = 14 + 7 = 21".to_string(),
                "Divide both sides by 3: x = 21 ÷ 3 = 7".to_string(),
                "Verify: 3(7) - 7 = 21 - 7 = 14 ✓".to_string(),
            ]),
            ("eq_006", "Solve: 4x + 8 = 2x + 18", "x = 5", vec![
                "Identify the equation: 4x + 8 = 2x + 18".to_string(),
                "Subtract 2x from both sides: 2x + 8 = 18".to_string(),
                "Subtract 8 from both sides: 2x = 10".to_string(),
                "Divide both sides by 2: x = 5".to_string(),
                "Verify: 4(5) + 8 = 28, 2(5) + 18 = 28 ✓".to_string(),
            ]),
        ];

        for (id, problem, answer, steps) in equation_problems {
            let difficulty = if id.ends_with("001") || id.ends_with("002") || id.ends_with("003") {
                DifficultyLevel::Elementary
            } else {
                DifficultyLevel::Intermediate
            };

            let math_problem = MathProblem::new(
                id.to_string(),
                problem.to_string(),
                MathProblemType::Equation,
                difficulty,
                answer.to_string(),
            )
            .with_steps(steps)
            .with_tags(vec!["equation".to_string(), "solving".to_string()]);

            self.add_problem(math_problem);
        }
    }

    /// Export dataset to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Load dataset from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get dataset statistics
    pub fn get_statistics(&self) -> DatasetStatistics {
        DatasetStatistics {
            total_problems: self.problems.len(),
            arithmetic_count: self.get_problems_by_type(&MathProblemType::Arithmetic).len(),
            algebra_count: self.get_problems_by_type(&MathProblemType::Algebra).len(),
            word_problem_count: self.get_problems_by_type(&MathProblemType::WordProblem).len(),
            equation_count: self.get_problems_by_type(&MathProblemType::Equation).len(),
            elementary_count: self.get_problems_by_difficulty(&DifficultyLevel::Elementary).len(),
            intermediate_count: self.get_problems_by_difficulty(&DifficultyLevel::Intermediate).len(),
            advanced_count: self.get_problems_by_difficulty(&DifficultyLevel::Advanced).len(),
        }
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub total_problems: usize,
    pub arithmetic_count: usize,
    pub algebra_count: usize,
    pub word_problem_count: usize,
    pub equation_count: usize,
    pub elementary_count: usize,
    pub intermediate_count: usize,
    pub advanced_count: usize,
}

impl DatasetStatistics {
    /// Format statistics for display
    pub fn format_statistics(&self) -> String {
        format!(
            "Dataset Statistics:\n\
             ==================\n\
             Total Problems: {}\n\n\
             By Type:\n\
             - Arithmetic: {}\n\
             - Algebra: {}\n\
             - Word Problems: {}\n\
             - Equations: {}\n\n\
             By Difficulty:\n\
             - Elementary: {}\n\
             - Intermediate: {}\n\
             - Advanced: {}\n",
            self.total_problems,
            self.arithmetic_count,
            self.algebra_count,
            self.word_problem_count,
            self.equation_count,
            self.elementary_count,
            self.intermediate_count,
            self.advanced_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_problem_creation() {
        let problem = MathProblem::new(
            "test_001".to_string(),
            "What is 2 + 2?".to_string(),
            MathProblemType::Arithmetic,
            DifficultyLevel::Elementary,
            "4".to_string(),
        );

        assert_eq!(problem.id, "test_001");
        assert_eq!(problem.problem_text, "What is 2 + 2?");
        assert_eq!(problem.expected_answer, "4");
    }

    #[test]
    fn test_dataset_creation() {
        let mut dataset = MathDataset::new(
            "Test Dataset".to_string(),
            "1.0.0".to_string(),
            "Test description".to_string(),
        );

        let problem = MathProblem::new(
            "test_001".to_string(),
            "What is 2 + 2?".to_string(),
            MathProblemType::Arithmetic,
            DifficultyLevel::Elementary,
            "4".to_string(),
        );

        dataset.add_problem(problem);
        assert_eq!(dataset.problems.len(), 1);
        assert_eq!(dataset.metadata.total_problems, 1);
    }

    #[test]
    fn test_default_dataset() {
        let dataset = MathDataset::create_default_dataset();
        assert!(dataset.problems.len() > 0);
        
        let stats = dataset.get_statistics();
        assert!(stats.arithmetic_count > 0);
        assert!(stats.algebra_count > 0);
        assert!(stats.word_problem_count > 0);
        assert!(stats.equation_count > 0);
    }

    #[test]
    fn test_problem_filtering() {
        let dataset = MathDataset::create_default_dataset();
        
        let arithmetic_problems = dataset.get_problems_by_type(&MathProblemType::Arithmetic);
        assert!(arithmetic_problems.len() > 0);
        
        let elementary_problems = dataset.get_problems_by_difficulty(&DifficultyLevel::Elementary);
        assert!(elementary_problems.len() > 0);
    }

    #[test]
    fn test_json_serialization() {
        let dataset = MathDataset::create_default_dataset();
        let json = dataset.to_json().unwrap();
        assert!(!json.is_empty());
        
        let deserialized = MathDataset::from_json(&json).unwrap();
        assert_eq!(dataset.problems.len(), deserialized.problems.len());
    }
}