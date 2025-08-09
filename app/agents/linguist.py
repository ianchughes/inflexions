"""Linguist Agent for refining category names and ensuring linguistic consistency."""

import logging
from typing import Dict, Any, List, Optional, Set
import re

from .base import BaseAgent
from ..models.puzzles import Puzzle, PuzzleCategory

logger = logging.getLogger(__name__)


class LinguistAgent(BaseAgent):
    """Linguist Agent that refines category names and ensures linguistic consistency."""
    
    def __init__(self, **kwargs):
        """Initialize the Linguist Agent."""
        super().__init__(**kwargs)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine category names and ensure linguistic consistency."""
        self.validate_input(input_data, ["puzzle"])
        
        try:
            puzzle_data = input_data["puzzle"]
            if isinstance(puzzle_data, dict):
                puzzle = Puzzle(**puzzle_data)
            else:
                puzzle = puzzle_data
            
            # Apply linguistic refinements
            refined_puzzle = self._refine_puzzle(puzzle)
            
            return {
                "success": True,
                "puzzle": refined_puzzle.dict(),
                "refinements": self._get_refinements_summary(puzzle, refined_puzzle),
                "metadata": self.get_agent_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in Linguist Agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": self.get_agent_metadata()
            }
    
    def _refine_puzzle(self, puzzle: Puzzle) -> Puzzle:
        """Apply linguistic refinements to the puzzle."""
        
        refined_puzzle = puzzle.copy(deep=True)
        refinements_made = []
        
        for category in refined_puzzle.solution:
            # Refine category name
            original_name = category.category_name
            refined_name = self._refine_category_name(category)
            
            if refined_name != original_name:
                category.category_name = refined_name
                refinements_made.append({
                    "type": "category_name_refinement",
                    "original": original_name,
                    "refined": refined_name
                })
            
            # Check linguistic consistency
            consistency_issues = self._check_linguistic_consistency(category)
            if consistency_issues:
                # Attempt to fix consistency issues
                fixed_words = self._fix_consistency_issues(category, consistency_issues)
                if fixed_words:
                    # Update words in both category and grid
                    self._update_words_in_puzzle(refined_puzzle, category.words, fixed_words)
                    category.words = fixed_words
                    
                    refinements_made.append({
                        "type": "linguistic_consistency_fix",
                        "category": category.category_name,
                        "issues": consistency_issues,
                        "original_words": category.words,
                        "fixed_words": fixed_words
                    })
            
            # Mark as processed by linguist
            category.processed_by_linguist = True
        
        # Update metadata
        if not refined_puzzle.generation_metadata.linguist_metadata:
            refined_puzzle.generation_metadata.linguist_metadata = {}
        
        refined_puzzle.generation_metadata.linguist_metadata.update({
            "refinements_made": refinements_made,
            "total_refinements": len(refinements_made)
        })
        
        return refined_puzzle
    
    def _refine_category_name(self, category: PuzzleCategory) -> str:
        """Refine a category name for conciseness, accuracy, and elegance."""
        
        words_list = self.format_word_list(category.words)
        
        prompt = self.create_system_prompt(
            "an expert editor and linguist specializing in puzzle creation",
            [
                "Refine category names to be concise, accurate, and elegant",
                "Ensure the name doesn't contain words from its own group",
                "Make names clear and unambiguous for puzzle solvers",
                "Maintain UK cultural authenticity where appropriate"
            ]
        )
        
        prompt += f"""
Refine the category name for this puzzle group:

Current name: "{category.category_name}"
Words in group: {words_list}
Category type: {category.category_type.value}

Guidelines:
1. The name should NOT contain any word that appears in the group
2. Make it concise but descriptive
3. Ensure it's culturally appropriate and accurate
4. Use proper capitalization and formatting
5. Avoid overly generic names

Respond with JSON:
{{
    "refined_name": "The improved category name",
    "reasoning": "Brief explanation of changes made"
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.3)
            parsed_response = self.parse_json_response(response)
            
            refined_name = parsed_response.get("refined_name", category.category_name)
            
            # Validate that refined name doesn't contain group words
            if self._name_contains_group_words(refined_name, category.words):
                logger.warning(f"Refined name '{refined_name}' contains group words, keeping original")
                return category.category_name
            
            return refined_name
            
        except Exception as e:
            logger.error(f"Error refining category name: {e}")
            return category.category_name
    
    def _check_linguistic_consistency(self, category: PuzzleCategory) -> List[Dict[str, Any]]:
        """Check for linguistic consistency issues within a category."""
        
        issues = []
        words = category.words
        
        # Check part of speech consistency
        pos_analysis = self._analyze_parts_of_speech(words)
        if pos_analysis["inconsistent"]:
            issues.append({
                "type": "part_of_speech_inconsistency",
                "description": "Mixed parts of speech in category",
                "details": pos_analysis
            })
        
        # Check number consistency (singular vs plural)
        number_analysis = self._analyze_number_consistency(words)
        if number_analysis["inconsistent"]:
            issues.append({
                "type": "number_inconsistency", 
                "description": "Mixed singular and plural forms",
                "details": number_analysis
            })
        
        # Check capitalization consistency
        capitalization_analysis = self._analyze_capitalization(words)
        if capitalization_analysis["inconsistent"]:
            issues.append({
                "type": "capitalization_inconsistency",
                "description": "Inconsistent capitalization patterns",
                "details": capitalization_analysis
            })
        
        # Check for obvious spelling/formatting issues
        formatting_issues = self._check_formatting_issues(words)
        if formatting_issues:
            issues.append({
                "type": "formatting_issues",
                "description": "Spelling or formatting problems",
                "details": formatting_issues
            })
        
        return issues
    
    def _analyze_parts_of_speech(self, words: List[str]) -> Dict[str, Any]:
        """Analyze parts of speech consistency."""
        
        words_list = self.format_word_list(words)
        
        prompt = f"""Analyze the parts of speech for these words: {words_list}

For each word, identify its primary part of speech (noun, verb, adjective, etc.) in the context of UK culture.

Respond with JSON:
{{
    "word_analysis": [
        {{"word": "word1", "pos": "noun", "confidence": "high"}},
        {{"word": "word2", "pos": "verb", "confidence": "medium"}},
        // ... etc
    ],
    "dominant_pos": "noun",
    "inconsistent": false,
    "mixed_pos_count": 1
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.1)
            parsed_response = self.parse_json_response(response)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error analyzing parts of speech: {e}")
            return {"inconsistent": False}
    
    def _analyze_number_consistency(self, words: List[str]) -> Dict[str, Any]:
        """Analyze singular/plural consistency."""
        
        words_list = self.format_word_list(words)
        
        prompt = f"""Analyze whether these words are consistently singular or plural: {words_list}

Consider UK English spelling conventions.

Respond with JSON:
{{
    "word_analysis": [
        {{"word": "word1", "number": "singular"}},
        {{"word": "word2", "number": "plural"}},
        // ... etc
    ],
    "dominant_number": "singular",
    "inconsistent": false,
    "mixed_count": 0
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.1)
            parsed_response = self.parse_json_response(response)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error analyzing number consistency: {e}")
            return {"inconsistent": False}
    
    def _analyze_capitalization(self, words: List[str]) -> Dict[str, Any]:
        """Analyze capitalization consistency."""
        
        # Simple analysis based on patterns
        title_case_count = sum(1 for word in words if word.istitle())
        upper_case_count = sum(1 for word in words if word.isupper())
        lower_case_count = sum(1 for word in words if word.islower())
        
        total_words = len(words)
        
        # Determine if there's inconsistency
        patterns = [title_case_count, upper_case_count, lower_case_count]
        max_pattern = max(patterns)
        
        # If no single pattern dominates (>= 75%), it's inconsistent
        inconsistent = max_pattern < (total_words * 0.75)
        
        return {
            "inconsistent": inconsistent,
            "title_case": title_case_count,
            "upper_case": upper_case_count, 
            "lower_case": lower_case_count,
            "dominant_pattern": "mixed" if inconsistent else (
                "title" if title_case_count == max_pattern else
                "upper" if upper_case_count == max_pattern else "lower"
            )
        }
    
    def _check_formatting_issues(self, words: List[str]) -> List[Dict[str, Any]]:
        """Check for formatting and spelling issues."""
        
        issues = []
        
        for word in words:
            # Check for obvious formatting problems
            if not word.strip():
                issues.append({"word": word, "issue": "empty_or_whitespace"})
            elif re.search(r'\s{2,}', word):
                issues.append({"word": word, "issue": "multiple_spaces"})
            elif word != word.strip():
                issues.append({"word": word, "issue": "leading_trailing_spaces"})
            elif re.search(r'[^\w\s\-\']', word):
                issues.append({"word": word, "issue": "unusual_characters"})
        
        return issues
    
    def _fix_consistency_issues(self, category: PuzzleCategory, issues: List[Dict[str, Any]]) -> Optional[List[str]]:
        """Attempt to fix linguistic consistency issues."""
        
        if not issues:
            return None
        
        words_list = self.format_word_list(category.words)
        issues_desc = [issue["description"] for issue in issues]
        
        prompt = f"""Fix linguistic consistency issues in this word group:

Words: {words_list}
Category: "{category.category_name}"
Issues to fix: {', '.join(issues_desc)}

Provide corrected versions of the words that:
1. Maintain the same meaning and cultural references
2. Fix the identified consistency issues
3. Keep all words as valid UK cultural references
4. Preserve the essential identity of each word

Only make minimal changes necessary to fix consistency issues.

Respond with JSON:
{{
    "fixed_words": ["corrected1", "corrected2", "corrected3", "corrected4"],
    "changes_made": ["description of changes"],
    "reasoning": "Brief explanation"
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.2)
            parsed_response = self.parse_json_response(response)
            
            fixed_words = parsed_response.get("fixed_words", [])
            
            # Validate that we still have exactly 4 words
            if len(fixed_words) == 4:
                return fixed_words
            else:
                logger.warning(f"Fixed words count mismatch: {len(fixed_words)} != 4")
                return None
                
        except Exception as e:
            logger.error(f"Error fixing consistency issues: {e}")
            return None
    
    def _update_words_in_puzzle(self, puzzle: Puzzle, old_words: List[str], new_words: List[str]) -> None:
        """Update words in the puzzle grid when they've been changed."""
        
        if len(old_words) != len(new_words):
            return
        
        word_mapping = dict(zip(old_words, new_words))
        
        # Update grid
        for i in range(4):
            for j in range(4):
                if puzzle.grid[i][j] in word_mapping:
                    puzzle.grid[i][j] = word_mapping[puzzle.grid[i][j]]
    
    def _name_contains_group_words(self, name: str, group_words: List[str]) -> bool:
        """Check if a category name contains any words from its group."""
        
        name_lower = name.lower()
        
        for word in group_words:
            word_lower = word.lower()
            # Check for exact word matches (not just substrings)
            if re.search(rf'\b{re.escape(word_lower)}\b', name_lower):
                return True
        
        return False
    
    def _get_refinements_summary(self, original_puzzle: Puzzle, refined_puzzle: Puzzle) -> Dict[str, Any]:
        """Generate a summary of linguistic refinements made."""
        
        refinements = []
        
        # Compare category names
        for i, (orig_cat, refined_cat) in enumerate(zip(original_puzzle.solution, refined_puzzle.solution)):
            if orig_cat.category_name != refined_cat.category_name:
                refinements.append({
                    "type": "category_name_change",
                    "category_index": i,
                    "original": orig_cat.category_name,
                    "refined": refined_cat.category_name
                })
            
            # Compare words
            if orig_cat.words != refined_cat.words:
                refinements.append({
                    "type": "word_consistency_fix",
                    "category_index": i,
                    "category_name": refined_cat.category_name,
                    "original_words": orig_cat.words,
                    "refined_words": refined_cat.words
                })
        
        return {
            "total_refinements": len(refinements),
            "refinements": refinements,
            "categories_modified": len(set(r.get("category_index") for r in refinements if "category_index" in r))
        }