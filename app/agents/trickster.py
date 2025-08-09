"""Trickster Agent for introducing deliberate misdirection to puzzles."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import random

from .base import BaseAgent
from ..models.puzzles import Puzzle, PuzzleCategory
from ..database.knowledge_graph import KnowledgeGraphDB
from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class TricksterAgent(BaseAgent):
    """Trickster Agent that introduces deliberate misdirection to make puzzles more challenging."""
    
    def __init__(self, knowledge_graph: KnowledgeGraphDB, vector_store: VectorStore, **kwargs):
        """Initialize the Trickster Agent."""
        super().__init__(**kwargs)
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add misdirection to a puzzle."""
        self.validate_input(input_data, ["puzzle"])
        
        try:
            puzzle_data = input_data["puzzle"]
            if isinstance(puzzle_data, dict):
                puzzle = Puzzle(**puzzle_data)
            else:
                puzzle = puzzle_data
            
            # Apply trickster modifications
            modified_puzzle = self._add_misdirection(puzzle)
            
            return {
                "success": True,
                "puzzle": modified_puzzle.dict(),
                "modifications": self._get_modifications_summary(puzzle, modified_puzzle),
                "metadata": self.get_agent_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in Trickster Agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": self.get_agent_metadata()
            }
    
    def _add_misdirection(self, puzzle: Puzzle) -> Puzzle:
        """Add misdirection to the puzzle by introducing red herrings."""
        
        # Find the most straightforward category (lowest difficulty/highest semantic similarity)
        target_category = self._identify_target_category(puzzle)
        
        if not target_category:
            logger.warning("No suitable target category found for misdirection")
            return puzzle
        
        # Find a red herring word that fits this category
        red_herring = self._find_red_herring(target_category, puzzle)
        
        if not red_herring:
            logger.warning("No suitable red herring found")
            return puzzle
        
        # Find a word to replace with the red herring
        replacement_target = self._find_replacement_target(puzzle, target_category, red_herring)
        
        if not replacement_target:
            logger.warning("No suitable replacement target found")
            return puzzle
        
        # Apply the modification
        modified_puzzle = self._apply_misdirection(puzzle, replacement_target, red_herring)
        
        # Mark categories as processed by trickster
        for category in modified_puzzle.solution:
            category.processed_by_trickster = True
        
        # Update metadata
        if not modified_puzzle.generation_metadata.trickster_metadata:
            modified_puzzle.generation_metadata.trickster_metadata = {}
        
        modified_puzzle.generation_metadata.trickster_metadata.update({
            "target_category": target_category.category_name,
            "red_herring_word": red_herring,
            "replaced_word": replacement_target["word"],
            "replacement_category": replacement_target["category"],
            "misdirection_applied": True
        })
        
        return modified_puzzle
    
    def _identify_target_category(self, puzzle: Puzzle) -> Optional[PuzzleCategory]:
        """Identify the most straightforward category for adding misdirection."""
        
        # Calculate semantic cohesion for each category
        category_scores = []
        
        for category in puzzle.solution:
            # Calculate semantic similarity using vector store
            cohesion_score = self.vector_store.calculate_category_cohesion(category.words)
            
            # Consider cultural specificity (lower tier = more straightforward)
            specificity_bonus = (5 - category.specificity_tier.value) * 0.1
            
            total_score = cohesion_score + specificity_bonus
            
            category_scores.append({
                "category": category,
                "score": total_score,
                "cohesion": cohesion_score
            })
        
        # Sort by score (highest = most straightforward)
        category_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Return the most straightforward category
        if category_scores:
            logger.info(f"Target category for misdirection: {category_scores[0]['category'].category_name} (score: {category_scores[0]['score']:.3f})")
            return category_scores[0]["category"]
        
        return None
    
    def _find_red_herring(self, target_category: PuzzleCategory, puzzle: Puzzle) -> Optional[str]:
        """Find a word that could serve as a red herring for the target category."""
        
        # Get all words already in the puzzle
        used_words = set()
        for category in puzzle.solution:
            used_words.update(category.words)
        
        # Try different strategies to find red herrings
        red_herring_candidates = []
        
        # Strategy 1: Use LLM to generate contextually similar words
        llm_candidates = self._generate_red_herrings_with_llm(target_category, used_words)
        red_herring_candidates.extend(llm_candidates)
        
        # Strategy 2: Use vector similarity to find related words
        vector_candidates = self._find_red_herrings_with_vectors(target_category, used_words)
        red_herring_candidates.extend(vector_candidates)
        
        # Strategy 3: Query knowledge graph for related entities
        kg_candidates = self._find_red_herrings_from_kg(target_category, used_words)
        red_herring_candidates.extend(kg_candidates)
        
        # Remove duplicates and used words
        unique_candidates = []
        seen = set()
        for candidate in red_herring_candidates:
            if candidate.lower() not in seen and candidate not in used_words:
                unique_candidates.append(candidate)
                seen.add(candidate.lower())
        
        # Return the best candidate
        if unique_candidates:
            # Use the first candidate for now - could add more sophisticated selection
            return unique_candidates[0]
        
        return None
    
    def _generate_red_herrings_with_llm(self, target_category: PuzzleCategory, used_words: set) -> List[str]:
        """Generate red herring candidates using LLM."""
        
        words_list = self.format_word_list(target_category.words)
        used_words_list = self.format_word_list(list(used_words))
        
        prompt = f"""Generate words that could be mistaken as belonging to the category "{target_category.category_name}".

Category words: {words_list}
Words to avoid (already used): {used_words_list}

Generate 5 words that:
1. Could plausibly fit the category at first glance
2. Have some connection to the theme but don't truly belong
3. Would create interesting misdirection for puzzle solvers
4. Are UK cultural references where appropriate

Respond with JSON:
{{
    "red_herrings": ["word1", "word2", "word3", "word4", "word5"],
    "reasoning": "Brief explanation of why these words could be misleading"
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.7)
            parsed_response = self.parse_json_response(response)
            
            red_herrings = parsed_response.get("red_herrings", [])
            return [word for word in red_herrings if isinstance(word, str) and len(word) > 0]
            
        except Exception as e:
            logger.error(f"Error generating red herrings with LLM: {e}")
            return []
    
    def _find_red_herrings_with_vectors(self, target_category: PuzzleCategory, used_words: set) -> List[str]:
        """Find red herring candidates using vector similarity."""
        
        try:
            # Get candidate words from the vector store
            candidates = self.vector_store.find_red_herring_candidates(
                target_category.words,
                [entity.entity_name for entity in self.vector_store.entities.values() 
                 if entity.entity_name not in used_words][:50]
            )
            
            # Return top candidates
            return [word for word, score in candidates[:5] if score > 0.3]
            
        except Exception as e:
            logger.error(f"Error finding red herrings with vectors: {e}")
            return []
    
    def _find_red_herrings_from_kg(self, target_category: PuzzleCategory, used_words: set) -> List[str]:
        """Find red herring candidates from the knowledge graph."""
        
        try:
            # Get entities of similar type/domain
            related_entities = self.knowledge_graph.get_entities_for_category(
                target_category.category_type.value, count=30
            )
            
            candidates = []
            for entity in related_entities:
                if (entity.entity_name not in used_words and 
                    entity.entity_name not in target_category.words):
                    candidates.append(entity.entity_name)
            
            return candidates[:5]
            
        except Exception as e:
            logger.error(f"Error finding red herrings from knowledge graph: {e}")
            return []
    
    def _find_replacement_target(self, puzzle: Puzzle, target_category: PuzzleCategory, red_herring: str) -> Optional[Dict[str, Any]]:
        """Find a word in a different category that can be replaced with the red herring."""
        
        # Look for words in other categories that are less central to their themes
        replacement_candidates = []
        
        for category in puzzle.solution:
            if category.category_name == target_category.category_name:
                continue
            
            for word in category.words:
                # Calculate how well this word fits its category
                other_words = [w for w in category.words if w != word]
                if len(other_words) >= 3:
                    # Calculate semantic distance from the word to its category
                    cohesion_without_word = self.vector_store.calculate_category_cohesion(other_words)
                    cohesion_with_word = self.vector_store.calculate_category_cohesion(category.words)
                    
                    # If removing this word doesn't hurt cohesion much, it's a good replacement target
                    impact = cohesion_with_word - cohesion_without_word
                    
                    replacement_candidates.append({
                        "word": word,
                        "category": category.category_name,
                        "impact": impact,
                        "category_obj": category
                    })
        
        # Sort by lowest impact (words that are least central to their categories)
        replacement_candidates.sort(key=lambda x: x["impact"])
        
        # Use LLM to validate the replacement makes sense
        if replacement_candidates:
            best_candidate = replacement_candidates[0]
            
            if self._validate_replacement(best_candidate, red_herring, target_category):
                return best_candidate
        
        return None
    
    def _validate_replacement(self, replacement_target: Dict[str, Any], red_herring: str, target_category: PuzzleCategory) -> bool:
        """Validate that a replacement creates good misdirection."""
        
        prompt = f"""Evaluate if this word replacement creates effective puzzle misdirection:

Original word: "{replacement_target['word']}" (in category: "{replacement_target['category']}")
Replacement word: "{red_herring}" (creates connection to: "{target_category.category_name}")

Target category words: {self.format_word_list(target_category.words)}

Would this replacement:
1. Create plausible misdirection (players might think "{red_herring}" belongs with the target category)?
2. Still allow the original category to be solvable?
3. Add strategic complexity without being unfair?

Respond with JSON:
{{
    "is_valid": true/false,
    "reasoning": "Brief explanation"
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.3)
            parsed_response = self.parse_json_response(response)
            
            return parsed_response.get("is_valid", False)
            
        except Exception as e:
            logger.error(f"Error validating replacement: {e}")
            return False
    
    def _apply_misdirection(self, puzzle: Puzzle, replacement_target: Dict[str, Any], red_herring: str) -> Puzzle:
        """Apply the misdirection by replacing a word in the puzzle."""
        
        # Create a new puzzle with the modification
        modified_puzzle = puzzle.copy(deep=True)
        
        # Find and replace the word in the grid
        target_word = replacement_target["word"]
        
        for i, row in enumerate(modified_puzzle.grid):
            for j, word in enumerate(row):
                if word == target_word:
                    modified_puzzle.grid[i][j] = red_herring
                    break
        
        # Update the solution categories
        for category in modified_puzzle.solution:
            if category.category_name == replacement_target["category"]:
                # Replace the word in this category
                new_words = [red_herring if word == target_word else word for word in category.words]
                category.words = new_words
                break
        
        return modified_puzzle
    
    def _get_modifications_summary(self, original_puzzle: Puzzle, modified_puzzle: Puzzle) -> Dict[str, Any]:
        """Generate a summary of modifications made."""
        
        modifications = []
        
        # Compare grids to find changes
        for i in range(4):
            for j in range(4):
                if original_puzzle.grid[i][j] != modified_puzzle.grid[i][j]:
                    modifications.append({
                        "type": "word_replacement",
                        "position": [i, j],
                        "original": original_puzzle.grid[i][j],
                        "modified": modified_puzzle.grid[i][j]
                    })
        
        return {
            "total_modifications": len(modifications),
            "changes": modifications,
            "misdirection_added": len(modifications) > 0
        }