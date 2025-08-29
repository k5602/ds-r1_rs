//! # Test Datasets for Evaluation
//!
//! Comprehensive test problem sets for evaluating reasoning capabilities.

use crate::utils::evaluation::{
    DifficultyLevel, ProblemCategory, ReasoningBenchmark, ReasoningProblem,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Collection of test datasets for different reasoning domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDatasetCollection {
    pub datasets: HashMap<String, ReasoningBenchmark>,
}

impl TestDatasetCollection {
    /// Create a new test dataset collection
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    /// Add a dataset to the collection
    pub fn add_dataset(&mut self, dataset: ReasoningBenchmark) {
        self.datasets.insert(dataset.name.clone(), dataset);
    }

    /// Get a dataset by name
    pub fn get_dataset(&self, name: &str) -> Option<&ReasoningBenchmark> {
        self.datasets.get(name)
    }

    /// Get all dataset names
    pub fn get_dataset_names(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }

    /// Get total number of problems across all datasets
    pub fn total_problems(&self) -> usize {
        self.datasets.values().map(|d| d.problems.len()).sum()
    }

    /// Get problems by category across all datasets
    pub fn get_problems_by_category(&self, category: &ProblemCategory) -> Vec<&ReasoningProblem> {
        self.datasets
            .values()
            .flat_map(|d| &d.problems)
            .filter(|p| &p.category == category)
            .collect()
    }

    /// Get problems by difficulty across all datasets
    pub fn get_problems_by_difficulty(
        &self,
        difficulty: &DifficultyLevel,
    ) -> Vec<&ReasoningProblem> {
        self.datasets
            .values()
            .flat_map(|d| &d.problems)
            .filter(|p| &p.difficulty == difficulty)
            .collect()
    }

    /// Create a comprehensive collection with all standard datasets
    pub fn create_comprehensive() -> Self {
        let mut collection = Self::new();

        // Add all standard benchmarks
        collection.add_dataset(create_extended_math_dataset());
        collection.add_dataset(create_extended_logic_dataset());
        collection.add_dataset(create_extended_programming_dataset());
        collection.add_dataset(create_code_explanation_dataset());
        collection.add_dataset(create_word_problems_dataset());
        collection.add_dataset(create_multi_step_reasoning_dataset());

        collection
    }

    /// Export datasets to JSON format
    pub fn export_to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Import datasets from JSON format
    pub fn import_from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Create an extended mathematical reasoning dataset
pub fn create_extended_math_dataset() -> ReasoningBenchmark {
    ReasoningBenchmark {
        name: "Extended Mathematical Reasoning".to_string(),
        problems: vec![
            // Basic Arithmetic
            ReasoningProblem {
                id: "ext_math_001".to_string(),
                problem_text: "Calculate: (15 + 25) × 2 - 10".to_string(),
                expected_answer: "70".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Easy,
            },
            ReasoningProblem {
                id: "ext_math_002".to_string(),
                problem_text: "What is 3/4 of 120?".to_string(),
                expected_answer: "90".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Easy,
            },
            ReasoningProblem {
                id: "ext_math_003".to_string(),
                problem_text: "If you buy 3 items at $4.50 each and pay with a $20 bill, how much change do you get?".to_string(),
                expected_answer: "$6.50".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Easy,
            },

            // Geometry
            ReasoningProblem {
                id: "ext_math_004".to_string(),
                problem_text: "A circle has a radius of 5 cm. What is its circumference? (Use π ≈ 3.14)".to_string(),
                expected_answer: "31.4 cm".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Medium,
            },
            ReasoningProblem {
                id: "ext_math_005".to_string(),
                problem_text: "A right triangle has legs of length 3 and 4. What is the length of the hypotenuse?".to_string(),
                expected_answer: "5".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Medium,
            },

            // Advanced Problems
            ReasoningProblem {
                id: "ext_math_006".to_string(),
                problem_text: "If f(x) = 2x + 3, what is f(f(2))?".to_string(),
                expected_answer: "17".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Hard,
            },
            ReasoningProblem {
                id: "ext_math_007".to_string(),
                problem_text: "A sequence starts with 1, 1, 2, 3, 5, 8, ... What is the 10th term?".to_string(),
                expected_answer: "55".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Hard,
            },
        ],
    }
}

/// Create an extended logical reasoning dataset
pub fn create_extended_logic_dataset() -> ReasoningBenchmark {
    ReasoningBenchmark {
        name: "Extended Logical Reasoning".to_string(),
        problems: vec![
            // Deductive Reasoning
            ReasoningProblem {
                id: "ext_logic_001".to_string(),
                problem_text: "All students in the class passed the test. Maria is a student in the class. Did Maria pass the test?".to_string(),
                expected_answer: "Yes".to_string(),
                category: ProblemCategory::Logic,
                difficulty: DifficultyLevel::Easy,
            },
            ReasoningProblem {
                id: "ext_logic_002".to_string(),
                problem_text: "No cats are dogs. Some pets are cats. Can we conclude that some pets are not dogs?".to_string(),
                expected_answer: "Yes".to_string(),
                category: ProblemCategory::Logic,
                difficulty: DifficultyLevel::Medium,
            },

            // Puzzle Logic
            ReasoningProblem {
                id: "ext_logic_003".to_string(),
                problem_text: "You have 3 boxes: one contains only apples, one contains only oranges, and one contains both. All boxes are labeled incorrectly. You can pick one fruit from one box. How can you correctly label all boxes?".to_string(),
                expected_answer: "Pick from the box labeled 'both' and use that information to deduce the others".to_string(),
                category: ProblemCategory::Logic,
                difficulty: DifficultyLevel::Hard,
            },
            ReasoningProblem {
                id: "ext_logic_004".to_string(),
                problem_text: "Five people are sitting in a row. Alice is not at either end. Bob is to the right of Alice. Carol is to the left of Alice. Where could Alice be sitting?".to_string(),
                expected_answer: "Position 2, 3, or 4".to_string(),
                category: ProblemCategory::Logic,
                difficulty: DifficultyLevel::Medium,
            },
        ],
    }
}

/// Create an extended programming dataset
pub fn create_extended_programming_dataset() -> ReasoningBenchmark {
    ReasoningBenchmark {
        name: "Extended Programming Reasoning".to_string(),
        problems: vec![
            // Algorithm Understanding
            ReasoningProblem {
                id: "ext_prog_001".to_string(),
                problem_text: "What is the output of this code?\n```\nlet mut x = 5;\nx = x * 2;\nx = x + 3;\nprintln!(\"{}\", x);\n```".to_string(),
                expected_answer: "13".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Easy,
            },
            ReasoningProblem {
                id: "ext_prog_002".to_string(),
                problem_text: "How many times will this loop execute?\n```\nfor i in 0..5 {\n    println!(\"{}\", i);\n}\n```".to_string(),
                expected_answer: "5 times".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Easy,
            },

            // Data Structure Analysis
            ReasoningProblem {
                id: "ext_prog_003".to_string(),
                problem_text: "What is the main advantage of using a hash map over an array for lookups?".to_string(),
                expected_answer: "Constant time O(1) average case lookup vs O(n) linear search".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Medium,
            },
            ReasoningProblem {
                id: "ext_prog_004".to_string(),
                problem_text: "When would you choose a linked list over an array?".to_string(),
                expected_answer: "When you need frequent insertions/deletions at arbitrary positions".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Medium,
            },

            // Algorithm Design
            ReasoningProblem {
                id: "ext_prog_005".to_string(),
                problem_text: "Describe an efficient algorithm to find the second largest element in an array.".to_string(),
                expected_answer: "Single pass keeping track of largest and second largest".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Hard,
            },
        ],
    }
}

/// Create a code explanation dataset
pub fn create_code_explanation_dataset() -> ReasoningBenchmark {
    ReasoningBenchmark {
        name: "Code Explanation and Analysis".to_string(),
        problems: vec![
            ReasoningProblem {
                id: "code_exp_001".to_string(),
                problem_text: "Explain what this function does step by step:\n```\nfn factorial(n: u32) -> u32 {\n    if n <= 1 {\n        1\n    } else {\n        n * factorial(n - 1)\n    }\n}\n```".to_string(),
                expected_answer: "Calculates factorial recursively by multiplying n with factorial of n-1".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Medium,
            },
            ReasoningProblem {
                id: "code_exp_002".to_string(),
                problem_text: "What potential issue exists in this code?\n```\nfn divide(a: f64, b: f64) -> f64 {\n    a / b\n}\n```".to_string(),
                expected_answer: "Division by zero when b is 0".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Easy,
            },
            ReasoningProblem {
                id: "code_exp_003".to_string(),
                problem_text: "Analyze the time complexity of this binary search implementation:\n```\nfn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let mut left = 0;\n    let mut right = arr.len();\n    while left < right {\n        let mid = (left + right) / 2;\n        if arr[mid] == target {\n            return Some(mid);\n        } else if arr[mid] < target {\n            left = mid + 1;\n        } else {\n            right = mid;\n        }\n    }\n    None\n}\n```".to_string(),
                expected_answer: "O(log n) because we halve the search space each iteration".to_string(),
                category: ProblemCategory::Programming,
                difficulty: DifficultyLevel::Hard,
            },
        ],
    }
}

/// Create a word problems dataset
pub fn create_word_problems_dataset() -> ReasoningBenchmark {
    ReasoningBenchmark {
        name: "Word Problems and Applied Reasoning".to_string(),
        problems: vec![
            ReasoningProblem {
                id: "word_001".to_string(),
                problem_text: "A store sells apples for $2 per pound and oranges for $3 per pound. If you buy 4 pounds of apples and 2 pounds of oranges, what is the total cost?".to_string(),
                expected_answer: "$14".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Easy,
            },
            ReasoningProblem {
                id: "word_002".to_string(),
                problem_text: "A car travels at 60 mph for 2 hours, then at 40 mph for 1 hour. What is the total distance traveled?".to_string(),
                expected_answer: "160 miles".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Medium,
            },
            ReasoningProblem {
                id: "word_003".to_string(),
                problem_text: "A recipe calls for 2 cups of flour for 12 cookies. How much flour is needed for 18 cookies?".to_string(),
                expected_answer: "3 cups".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Medium,
            },
            ReasoningProblem {
                id: "word_004".to_string(),
                problem_text: "A tank can be filled by pipe A in 6 hours and by pipe B in 4 hours. If both pipes work together, how long will it take to fill the tank?".to_string(),
                expected_answer: "2.4 hours".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Hard,
            },
        ],
    }
}

/// Create a multi-step reasoning dataset
pub fn create_multi_step_reasoning_dataset() -> ReasoningBenchmark {
    ReasoningBenchmark {
        name: "Multi-Step Reasoning Challenges".to_string(),
        problems: vec![
            ReasoningProblem {
                id: "multi_001".to_string(),
                problem_text: "A company has 100 employees. 60% work in sales, 30% work in engineering, and the rest work in administration. If the sales team gets a 10% bonus and the engineering team gets a 15% bonus, what percentage of all employees get a bonus?".to_string(),
                expected_answer: "90%".to_string(),
                category: ProblemCategory::General,
                difficulty: DifficultyLevel::Medium,
            },
            ReasoningProblem {
                id: "multi_002".to_string(),
                problem_text: "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons of water?".to_string(),
                expected_answer: "Fill 5-gallon jug, pour into 3-gallon jug, empty 3-gallon jug, pour remaining 2 gallons from 5-gallon jug into 3-gallon jug, fill 5-gallon jug again, pour into 3-gallon jug until full (1 gallon), leaving 4 gallons in 5-gallon jug".to_string(),
                category: ProblemCategory::Logic,
                difficulty: DifficultyLevel::Hard,
            },
            ReasoningProblem {
                id: "multi_003".to_string(),
                problem_text: "A farmer has chickens and cows. In total, there are 30 heads and 74 legs. How many chickens and how many cows are there?".to_string(),
                expected_answer: "23 chickens and 7 cows".to_string(),
                category: ProblemCategory::Mathematics,
                difficulty: DifficultyLevel::Hard,
            },
        ],
    }
}

impl Default for TestDatasetCollection {
    fn default() -> Self {
        Self::create_comprehensive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_collection_creation() {
        let collection = TestDatasetCollection::new();
        assert_eq!(collection.datasets.len(), 0);
        assert_eq!(collection.total_problems(), 0);
    }

    #[test]
    fn test_comprehensive_collection() {
        let collection = TestDatasetCollection::create_comprehensive();
        assert!(collection.datasets.len() > 0);
        assert!(collection.total_problems() > 0);

        let names = collection.get_dataset_names();
        assert!(names.contains(&"Extended Mathematical Reasoning".to_string()));
        assert!(names.contains(&"Extended Logical Reasoning".to_string()));
    }

    #[test]
    fn test_problems_by_category() {
        let collection = TestDatasetCollection::create_comprehensive();
        let math_problems = collection.get_problems_by_category(&ProblemCategory::Mathematics);
        let logic_problems = collection.get_problems_by_category(&ProblemCategory::Logic);
        let prog_problems = collection.get_problems_by_category(&ProblemCategory::Programming);

        assert!(!math_problems.is_empty());
        assert!(!logic_problems.is_empty());
        assert!(!prog_problems.is_empty());
    }

    #[test]
    fn test_problems_by_difficulty() {
        let collection = TestDatasetCollection::create_comprehensive();
        let easy_problems = collection.get_problems_by_difficulty(&DifficultyLevel::Easy);
        let medium_problems = collection.get_problems_by_difficulty(&DifficultyLevel::Medium);
        let hard_problems = collection.get_problems_by_difficulty(&DifficultyLevel::Hard);

        assert!(!easy_problems.is_empty());
        assert!(!medium_problems.is_empty());
        assert!(!hard_problems.is_empty());
    }

    #[test]
    fn test_json_serialization() {
        let collection = TestDatasetCollection::create_comprehensive();
        let json = collection.export_to_json().unwrap();
        assert!(!json.is_empty());

        let deserialized = TestDatasetCollection::import_from_json(&json).unwrap();
        assert_eq!(collection.datasets.len(), deserialized.datasets.len());
    }

    #[test]
    fn test_individual_datasets() {
        let math_dataset = create_extended_math_dataset();
        assert_eq!(math_dataset.name, "Extended Mathematical Reasoning");
        assert!(!math_dataset.problems.is_empty());

        let logic_dataset = create_extended_logic_dataset();
        assert_eq!(logic_dataset.name, "Extended Logical Reasoning");
        assert!(!logic_dataset.problems.is_empty());

        let prog_dataset = create_extended_programming_dataset();
        assert_eq!(prog_dataset.name, "Extended Programming Reasoning");
        assert!(!prog_dataset.problems.is_empty());
    }
}
