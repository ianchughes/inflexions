"""Judge Agent for difficulty assessment with hybrid quantitative and qualitative scoring."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import Counter

from .base import BaseAgent
from ..models.puzzles import Puzzle, PuzzleCategory, DifficultyLevel, DifficultyMetrics
from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """Judge Agent that assigns final difficulty ratings using hybrid assessment."""
    
    def __init__(self, vector_store: VectorStore, **kwargs):
        """Initialize the Judge Agent."""
        super().__init__(**kwargs)
        self.vector_store = vector_store
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assign difficulty ratings to puzzle categories."""
        self.validate_input(input_data, ["puzzle"])
        
        try:
            puzzle_data = input_data["puzzle"]
            if isinstance(puzzle_data, dict):
                puzzle = Puzzle(**puzzle_data)
            else:
                puzzle = puzzle_data
            
            # Assess difficulty for each category
            assessed_puzzle = self._assess_puzzle_difficulty(puzzle)
            
            return {
                "success": True,
                "puzzle": assessed_puzzle.dict(),
                "difficulty_analysis": self._get_difficulty_analysis(assessed_puzzle),
                "metadata": self.get_agent_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in Judge Agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": self.get_agent_metadata()
            }
    
    def _assess_puzzle_difficulty(self, puzzle: Puzzle) -> Puzzle:
        """Assess difficulty for all categories in the puzzle."""
        
        assessed_puzzle = puzzle.copy(deep=True)
        
        # Step 1: Calculate quantitative metrics for each category
        quantitative_scores = []
        for category in assessed_puzzle.solution:
            metrics = self._calculate_quantitative_metrics(category)
            category.difficulty_metrics = metrics
            quantitative_scores.append({
                "category": category,
                "metrics": metrics,
                "composite_score": self._calculate_composite_score(metrics)
            })
        
        # Step 2: Use LLM for qualitative assessment
        qualitative_assessment = self._perform_qualitative_assessment(assessed_puzzle, quantitative_scores)
        
        # Step 3: Assign final difficulty levels
        self._assign_final_difficulties(assessed_puzzle, quantitative_scores, qualitative_assessment)
        
        # Step 4: Validate difficulty distribution
        self._validate_and_adjust_difficulty_distribution(assessed_puzzle)
        
        # Update metadata
        if not assessed_puzzle.generation_metadata.judge_metadata:
            assessed_puzzle.generation_metadata.judge_metadata = {}
        
        assessed_puzzle.generation_metadata.judge_metadata.update({
            "assessment_model": self.model_name,
            "quantitative_scores": [{"category": score["category"].category_name, "score": score["composite_score"]} for score in quantitative_scores],
            "qualitative_assessment": qualitative_assessment,
            "final_difficulty_distribution": self._get_difficulty_distribution(assessed_puzzle)
        })
        
        return assessed_puzzle
    
    def _calculate_quantitative_metrics(self, category: PuzzleCategory) -> DifficultyMetrics:
        """Calculate quantitative metrics for difficulty assessment."""
        
        # 1. Semantic Cohesion (using vector store)
        cosine_similarity = self.vector_store.calculate_category_cohesion(category.words)
        
        # 2. Word Frequency Analysis
        word_frequency = self._analyze_word_frequency(category.words)
        
        # 3. Polysemy Score (number of meanings per word)
        polysemy_score = self._calculate_polysemy_score(category.words)
        
        # 4. Cultural Specificity Variance
        cultural_variance = self._calculate_cultural_specificity_variance(category)
        
        return DifficultyMetrics(
            cosine_similarity=cosine_similarity,
            word_frequency=word_frequency,
            polysemy_score=polysemy_score,
            cultural_specificity_variance=cultural_variance
        )
    
    def _analyze_word_frequency(self, words: List[str]) -> str:
        """Analyze word frequency rating (high/medium/low)."""
        
        # Use LLM to assess word frequency in UK English
        words_list = self.format_word_list(words)
        
        prompt = f"""Analyze the frequency/commonality of these words in everyday UK English: {words_list}

Rate the overall frequency level of this word group as:
- "high": Common words that most UK English speakers would know
- "medium": Moderately common words, known by most but not universal  
- "low": Uncommon or specialized words requiring specific knowledge

Respond with JSON:
{{
    "frequency_rating": "high" | "medium" | "low",
    "word_analysis": [
        {{"word": "word1", "frequency": "high", "reasoning": "common everyday word"}},
        // ... for each word
    ],
    "overall_reasoning": "Brief explanation"
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.2)
            parsed_response = self.parse_json_response(response)
            
            return parsed_response.get("frequency_rating", "medium")
            
        except Exception as e:
            logger.error(f"Error analyzing word frequency: {e}")
            return "medium"
    
    def _calculate_polysemy_score(self, words: List[str]) -> float:
        """Calculate average polysemy score (number of meanings per word)."""
        
        words_list = self.format_word_list(words)
        
        prompt = f"""Analyze the polysemy (number of distinct meanings) for each word: {words_list}

For each word, count how many significantly different meanings it has in UK English.

Respond with JSON:
{{
    "word_polysemy": [
        {{"word": "word1", "meaning_count": 3, "examples": ["meaning 1", "meaning 2", "meaning 3"]}},
        // ... for each word
    ],
    "average_polysemy": 2.5
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.2)
            parsed_response = self.parse_json_response(response)
            
            return parsed_response.get("average_polysemy", 2.0)
            
        except Exception as e:
            logger.error(f"Error calculating polysemy score: {e}")
            return 2.0
    
    def _calculate_cultural_specificity_variance(self, category: PuzzleCategory) -> float:
        """Calculate variance in cultural specificity among words."""
        
        # If all words have the same cultural tier, variance is 0
        # If there's a mix, calculate the variance
        
        # For now, use the category's overall tier
        # In a more sophisticated implementation, each word might have its own tier
        return 0.0
    
    def _calculate_composite_score(self, metrics: DifficultyMetrics) -> float:
        """Calculate a composite difficulty score from quantitative metrics."""
        
        # Convert frequency rating to numeric score
        frequency_scores = {"high": 0.2, "medium": 0.5, "low": 0.8}
        frequency_score = frequency_scores.get(metrics.word_frequency, 0.5)
        
        # Inverse relationship: higher similarity = easier (lower difficulty)
        similarity_score = 1.0 - metrics.cosine_similarity
        
        # Higher polysemy = harder
        polysemy_score = min(1.0, (metrics.polysemy_score or 2.0) / 5.0)
        
        # Cultural variance score
        variance_score = metrics.cultural_specificity_variance or 0.0
        
        # Weighted composite score
        composite = (
            similarity_score * 0.4 +      # Semantic cohesion (40%)
            frequency_score * 0.3 +       # Word frequency (30%)
            polysemy_score * 0.2 +        # Polysemy (20%)
            variance_score * 0.1          # Cultural variance (10%)
        )
        
        return min(1.0, max(0.0, composite))
    
    def _perform_qualitative_assessment(self, puzzle: Puzzle, quantitative_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform qualitative assessment using LLM expertise."""
        
        # Format quantitative data for the prompt
        categories_info = []
        for score_data in quantitative_scores:
            category = score_data["category"]
            metrics = score_data["metrics"]
            composite = score_data["composite_score"]
            
            words_list = self.format_word_list(category.words)
            
            category_info = f"""
Category: "{category.category_name}"
Words: {words_list}
Type: {category.category_type.value}
Cultural Tier: {category.specificity_tier.value}
Quantitative Metrics:
- Semantic Cohesion: {metrics.cosine_similarity:.3f}
- Word Frequency: {metrics.word_frequency}
- Polysemy Score: {metrics.polysemy_score:.2f}
- Composite Score: {composite:.3f}"""
            
            categories_info.append(category_info)
        
        categories_text = "\n\n".join(categories_info)
        
        prompt = self.create_system_prompt(
            "an expert puzzle editor and difficulty assessor",
            [
                "Assess puzzle difficulty using deep understanding of cognitive complexity",
                "Consider the type of mental leap required for each category",
                "Factor in cultural knowledge requirements and misdirection potential",
                "Assign difficulty colors: Yellow (easiest), Green, Blue, Purple (hardest)"
            ]
        )
        
        prompt += f"""
Analyze these puzzle categories and assign difficulty levels:

{categories_text}

Consider for each category:
1. What type of cognitive process is required? (recognition, recall, inference, wordplay)
2. How much UK cultural knowledge is needed?
3. How obvious or subtle is the connection?
4. Could this category mislead players initially?

Difficulty Levels:
- YELLOW: Straightforward connections, common knowledge
- GREEN: Clear connections requiring some thought or cultural knowledge  
- BLUE: More complex connections requiring deeper cultural knowledge or inference
- PURPLE: Most challenging - abstract connections, wordplay, or highly specific knowledge

Respond with JSON:
{{
    "category_assessments": [
        {{
            "category_name": "Category Name",
            "suggested_difficulty": "YELLOW" | "GREEN" | "BLUE" | "PURPLE",
            "cognitive_complexity": "description of mental process required",
            "cultural_knowledge_required": "assessment of UK cultural knowledge needed",
            "misdirection_potential": "how this might mislead players",
            "justification": "detailed reasoning for difficulty assignment"
        }},
        // ... for each category
    ],
    "overall_assessment": "analysis of the puzzle's difficulty balance"
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.3)
            parsed_response = self.parse_json_response(response)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error in qualitative assessment: {e}")
            return {"category_assessments": [], "overall_assessment": "Assessment failed"}
    
    def _assign_final_difficulties(self, puzzle: Puzzle, quantitative_scores: List[Dict[str, Any]], qualitative_assessment: Dict[str, Any]) -> None:
        """Assign final difficulty levels combining quantitative and qualitative assessments."""
        
        qualitative_categories = {
            assessment["category_name"]: assessment 
            for assessment in qualitative_assessment.get("category_assessments", [])
        }
        
        category_difficulty_data = []
        
        for score_data in quantitative_scores:
            category = score_data["category"]
            composite_score = score_data["composite_score"]
            
            # Get qualitative assessment
            qual_assessment = qualitative_categories.get(category.category_name, {})
            suggested_difficulty = qual_assessment.get("suggested_difficulty")
            
            # Combine quantitative and qualitative
            final_difficulty = self._determine_final_difficulty(
                composite_score, 
                suggested_difficulty, 
                category
            )
            
            category.difficulty = final_difficulty
            category.justification = qual_assessment.get("justification", "Difficulty assigned based on quantitative metrics")
            
            category_difficulty_data.append({
                "category": category,
                "quantitative_score": composite_score,
                "qualitative_suggestion": suggested_difficulty,
                "final_difficulty": final_difficulty
            })
    
    def _determine_final_difficulty(self, quantitative_score: float, qualitative_suggestion: Optional[str], category: PuzzleCategory) -> DifficultyLevel:
        """Determine final difficulty by combining quantitative and qualitative assessments."""
        
        # Map quantitative score to difficulty ranges
        if quantitative_score < 0.25:
            quantitative_difficulty = DifficultyLevel.YELLOW
        elif quantitative_score < 0.5:
            quantitative_difficulty = DifficultyLevel.GREEN
        elif quantitative_score < 0.75:
            quantitative_difficulty = DifficultyLevel.BLUE
        else:
            quantitative_difficulty = DifficultyLevel.PURPLE
        
        # If we have a qualitative suggestion, give it priority
        if qualitative_suggestion:
            try:
                qualitative_difficulty = DifficultyLevel(qualitative_suggestion)
                
                # If they agree, use it
                if quantitative_difficulty == qualitative_difficulty:
                    return qualitative_difficulty
                
                # If they disagree by one level, average them
                difficulty_order = [DifficultyLevel.YELLOW, DifficultyLevel.GREEN, DifficultyLevel.BLUE, DifficultyLevel.PURPLE]
                quant_index = difficulty_order.index(quantitative_difficulty)
                qual_index = difficulty_order.index(qualitative_difficulty)
                
                if abs(quant_index - qual_index) == 1:
                    # Take the higher difficulty (more conservative)
                    final_index = max(quant_index, qual_index)
                    return difficulty_order[final_index]
                else:
                    # Significant disagreement - favor qualitative assessment
                    return qualitative_difficulty
                    
            except ValueError:
                logger.warning(f"Invalid qualitative difficulty suggestion: {qualitative_suggestion}")
        
        return quantitative_difficulty
    
    def _validate_and_adjust_difficulty_distribution(self, puzzle: Puzzle) -> None:
        """Ensure we have a valid difficulty distribution (one of each color)."""
        
        current_difficulties = [category.difficulty for category in puzzle.solution]
        difficulty_counts = Counter(current_difficulties)
        
        # Check if we have exactly one of each difficulty
        expected_difficulties = [DifficultyLevel.YELLOW, DifficultyLevel.GREEN, DifficultyLevel.BLUE, DifficultyLevel.PURPLE]
        
        # If we already have one of each, we're done
        if all(difficulty_counts[diff] == 1 for diff in expected_difficulties):
            return
        
        # Otherwise, adjust to ensure one of each
        self._rebalance_difficulties(puzzle, difficulty_counts)
    
    def _rebalance_difficulties(self, puzzle: Puzzle, current_counts: Counter) -> None:
        """Rebalance difficulties to ensure one of each color."""
        
        target_difficulties = [DifficultyLevel.YELLOW, DifficultyLevel.GREEN, DifficultyLevel.BLUE, DifficultyLevel.PURPLE]
        
        # Identify which difficulties are missing or duplicated
        missing = [diff for diff in target_difficulties if current_counts[diff] == 0]
        duplicated = [diff for diff in target_difficulties if current_counts[diff] > 1]
        
        if not missing:
            return
        
        # Sort categories by their quantitative scores to make informed adjustments
        categories_with_scores = []
        for category in puzzle.solution:
            composite_score = 0.5  # Default if no metrics
            if category.difficulty_metrics:
                composite_score = self._calculate_composite_score(category.difficulty_metrics)
            categories_with_scores.append((category, composite_score))
        
        categories_with_scores.sort(key=lambda x: x[1])
        
        # Reassign difficulties based on sorted order
        for i, (category, score) in enumerate(categories_with_scores):
            category.difficulty = target_difficulties[i]
            logger.info(f"Rebalanced {category.category_name} to {target_difficulties[i].value}")
    
    def _get_difficulty_distribution(self, puzzle: Puzzle) -> Dict[str, int]:
        """Get the distribution of difficulty levels in the puzzle."""
        
        distribution = Counter()
        for category in puzzle.solution:
            if category.difficulty:
                distribution[category.difficulty.value] += 1
        
        return dict(distribution)
    
    def _get_difficulty_analysis(self, puzzle: Puzzle) -> Dict[str, Any]:
        """Generate analysis of the difficulty assessment."""
        
        categories_analysis = []
        
        for category in puzzle.solution:
            analysis = {
                "category_name": category.category_name,
                "assigned_difficulty": category.difficulty.value if category.difficulty else None,
                "justification": category.justification,
                "quantitative_metrics": category.difficulty_metrics.dict() if category.difficulty_metrics else None
            }
            categories_analysis.append(analysis)
        
        difficulty_distribution = self._get_difficulty_distribution(puzzle)
        
        return {
            "categories": categories_analysis,
            "difficulty_distribution": difficulty_distribution,
            "is_balanced": all(count == 1 for count in difficulty_distribution.values()),
            "total_categories": len(puzzle.solution)
        }