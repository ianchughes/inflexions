"""Creator Agent for puzzle generation using Tree of Thoughts prompting."""

import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from .base import BaseAgent
from ..models.puzzles import (
    Puzzle, PuzzleCategory, GenerationConstraints, GenerationMetadata, 
    CategoryType, DifficultyLevel, PuzzleStatus
)
from ..models.entities import CulturalSpecificityTier
from ..database.knowledge_graph import KnowledgeGraphDB
from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class CreatorAgent(BaseAgent):
    """Creator Agent that generates puzzles using Tree of Thoughts prompting."""
    
    def __init__(self, knowledge_graph: KnowledgeGraphDB, vector_store: VectorStore, **kwargs):
        """Initialize the Creator Agent."""
        super().__init__(**kwargs)
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a puzzle using Tree of Thoughts methodology."""
        self.validate_input(input_data, ["constraints"])
        
        constraints = input_data.get("constraints", {})
        if isinstance(constraints, dict):
            constraints = GenerationConstraints(**constraints)
        
        try:
            # Tree of Thoughts: Multi-step reasoning process
            puzzle = self._generate_puzzle_with_tot(constraints)
            
            return {
                "success": True,
                "puzzle": puzzle.dict(),
                "metadata": self.get_agent_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in Creator Agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": self.get_agent_metadata()
            }
    
    def _generate_puzzle_with_tot(self, constraints: GenerationConstraints) -> Puzzle:
        """Generate puzzle using Tree of Thoughts prompting strategy."""
        
        # Step 1: Theme Brainstorming
        themes = self._brainstorm_themes(constraints)
        logger.info(f"Generated {len(themes)} potential themes")
        
        # Step 2: Word Grid Expansion
        candidate_grids = []
        for theme_set in themes[:3]:  # Use top 3 theme sets
            grid = self._generate_word_grid(theme_set, constraints)
            if grid:
                candidate_grids.append(grid)
        
        if not candidate_grids:
            raise ValueError("Failed to generate any valid word grids")
        
        # Step 3: Self-Evaluation & Selection
        best_grid = self._evaluate_and_select_grid(candidate_grids)
        
        # Step 4: Create puzzle object
        puzzle = self._create_puzzle_object(best_grid, constraints)
        
        return puzzle
    
    def _brainstorm_themes(self, constraints: GenerationConstraints) -> List[List[Dict[str, Any]]]:
        """Step 1: Brainstorm potential themes for puzzle categories."""
        
        prompt = self.create_system_prompt(
            "an expert puzzle creator specializing in UK cultural connections",
            [
                "Generate diverse, culturally relevant themes for UK Connections puzzles",
                "Each theme should be specific enough to have exactly 4 related items",
                "Themes should span different cultural domains (TV, history, food, etc.)",
                "Consider different connection types (semantic, cultural, wordplay, etc.)"
            ]
        )
        
        # Add constraint information to prompt
        constraint_info = self._format_constraints_for_prompt(constraints)
        
        prompt += f"""
Generate 6 different theme sets for a UK Connections puzzle. Each theme set should contain 4 distinct categories.

{constraint_info}

For each theme set, provide:
1. Four category themes with their connection type
2. Brief justification for cultural relevance
3. Estimated difficulty tier (1-4)

Respond in JSON format:
{{
    "theme_sets": [
        {{
            "categories": [
                {{
                    "theme": "Characters from Fawlty Towers",
                    "connection_type": "CULTURAL_TV",
                    "tier": 3,
                    "justification": "Classic UK sitcom with memorable characters"
                }},
                // ... 3 more categories
            ]
        }},
        // ... 5 more theme sets
    ]
}}
"""
        
        response = self.call_llm_with_cache(prompt, temperature=0.9)
        
        try:
            parsed_response = self.parse_json_response(response)
            theme_sets = parsed_response.get("theme_sets", [])
            
            # Convert to internal format
            formatted_themes = []
            for theme_set in theme_sets:
                categories = theme_set.get("categories", [])
                if len(categories) == 4:
                    formatted_themes.append(categories)
            
            return formatted_themes
            
        except Exception as e:
            logger.error(f"Error parsing theme response: {e}")
            # Fallback to default themes
            return self._get_fallback_themes()
    
    def _generate_word_grid(self, theme_set: List[Dict[str, Any]], constraints: GenerationConstraints) -> Optional[Dict[str, Any]]:
        """Step 2: Generate word grid for a specific theme set."""
        
        categories = []
        all_words = []
        
        for theme in theme_set:
            # Get words for this theme from knowledge graph
            words = self._get_words_for_theme(theme, constraints)
            
            if len(words) < 4:
                logger.warning(f"Insufficient words for theme: {theme['theme']}")
                return None
            
            # Take first 4 words
            category_words = words[:4]
            categories.append({
                "theme": theme,
                "words": category_words
            })
            all_words.extend(category_words)
        
        if len(all_words) != 16:
            logger.warning(f"Invalid word count: {len(all_words)}")
            return None
        
        # Check for word uniqueness
        if len(set(all_words)) != 16:
            logger.warning("Duplicate words in grid")
            return None
        
        # Shuffle words into 4x4 grid
        shuffled_words = all_words.copy()
        random.shuffle(shuffled_words)
        
        grid = [
            shuffled_words[0:4],
            shuffled_words[4:8], 
            shuffled_words[8:12],
            shuffled_words[12:16]
        ]
        
        return {
            "grid": grid,
            "categories": categories,
            "all_words": all_words
        }
    
    def _evaluate_and_select_grid(self, candidate_grids: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 3: Evaluate candidate grids and select the best one."""
        
        if len(candidate_grids) == 1:
            return candidate_grids[0]
        
        # Create evaluation prompt
        prompt = self.create_system_prompt(
            "an expert puzzle evaluator",
            [
                "Evaluate puzzle grids for quality, distinctness, and player engagement",
                "Consider thematic variety, difficulty balance, and potential for misdirection",
                "Select the grid with the best overall puzzle quality"
            ]
        )
        
        # Format candidate grids for evaluation
        grid_descriptions = []
        for i, grid_data in enumerate(candidate_grids):
            categories_desc = []
            for cat in grid_data["categories"]:
                theme = cat["theme"]
                words = self.format_word_list(cat["words"])
                categories_desc.append(f"- {theme['theme']}: {words}")
            
            grid_desc = f"Grid {i+1}:\n" + "\n".join(categories_desc)
            grid_descriptions.append(grid_desc)
        
        prompt += f"""
Evaluate these {len(candidate_grids)} puzzle grid candidates and select the best one:

{chr(10).join(grid_descriptions)}

Consider:
1. Thematic distinctness between categories
2. Appropriate difficulty progression
3. Cultural relevance and authenticity
4. Potential for interesting misdirection

Respond with JSON:
{{
    "selected_grid": 1,  // Grid number (1-{len(candidate_grids)})
    "reasoning": "Explanation for selection"
}}
"""
        
        response = self.call_llm_with_cache(prompt, temperature=0.3)
        
        try:
            parsed_response = self.parse_json_response(response)
            selected_index = parsed_response.get("selected_grid", 1) - 1
            
            # Validate selection
            if 0 <= selected_index < len(candidate_grids):
                return candidate_grids[selected_index]
            else:
                logger.warning("Invalid grid selection, using first grid")
                return candidate_grids[0]
                
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return candidate_grids[0]
    
    def _get_words_for_theme(self, theme: Dict[str, Any], constraints: GenerationConstraints) -> List[str]:
        """Get words from knowledge graph for a specific theme."""
        
        theme_name = theme["theme"]
        connection_type = theme.get("connection_type", "OTHER")
        
        # Query knowledge graph based on theme type
        if connection_type in ["CULTURAL_TV", "CULTURAL_FILM", "CULTURAL_MUSIC", "CULTURAL_LITERATURE"]:
            words = self._get_cultural_words(theme_name, connection_type)
        elif connection_type == "GEOGRAPHICAL":
            words = self._get_geographical_words(theme_name)
        elif connection_type == "CULINARY":
            words = self._get_culinary_words(theme_name)
        else:
            # Use LLM to generate words for other categories
            words = self._generate_words_with_llm(theme_name, connection_type)
        
        return words
    
    def _get_cultural_words(self, theme_name: str, connection_type: str) -> List[str]:
        """Get words for cultural themes from knowledge graph."""
        try:
            # Extract category from theme name and query knowledge graph
            entities = self.knowledge_graph.get_entities_for_category(connection_type, count=20)
            
            # Filter entities based on theme specifics
            relevant_entities = []
            for entity in entities:
                if self._entity_matches_theme(entity, theme_name):
                    relevant_entities.append(entity.entity_name)
            
            return relevant_entities[:10]  # Return up to 10 candidates
            
        except Exception as e:
            logger.error(f"Error getting cultural words: {e}")
            return []
    
    def _get_geographical_words(self, theme_name: str) -> List[str]:
        """Get geographical words from knowledge graph."""
        try:
            from ..models.entities import EntityType
            entities = self.knowledge_graph.get_entities_by_type(EntityType.LOCATION, limit=20)
            
            # Filter based on theme specifics
            words = []
            for entity in entities:
                if self._entity_matches_theme(entity, theme_name):
                    words.append(entity.entity_name)
            
            return words[:10]
            
        except Exception as e:
            logger.error(f"Error getting geographical words: {e}")
            return []
    
    def _get_culinary_words(self, theme_name: str) -> List[str]:
        """Get culinary words from knowledge graph."""
        try:
            from ..models.entities import EntityType
            food_entities = self.knowledge_graph.get_entities_by_type(EntityType.FOOD, limit=15)
            drink_entities = self.knowledge_graph.get_entities_by_type(EntityType.DRINK, limit=15)
            
            all_entities = food_entities + drink_entities
            
            words = []
            for entity in all_entities:
                if self._entity_matches_theme(entity, theme_name):
                    words.append(entity.entity_name)
            
            return words[:10]
            
        except Exception as e:
            logger.error(f"Error getting culinary words: {e}")
            return []
    
    def _generate_words_with_llm(self, theme_name: str, connection_type: str) -> List[str]:
        """Generate words using LLM for themes not in knowledge graph."""
        
        prompt = f"""Generate exactly 6 words that fit the theme: "{theme_name}"

Connection type: {connection_type}

Requirements:
- All words must be UK cultural references where appropriate
- Words should be single terms or proper nouns
- No phrases or multi-word expressions
- Ensure cultural authenticity

Respond with JSON:
{{
    "words": ["word1", "word2", "word3", "word4", "word5", "word6"]
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.5)
            parsed_response = self.parse_json_response(response)
            words = parsed_response.get("words", [])
            
            # Validate words
            valid_words = [word for word in words if isinstance(word, str) and len(word) > 0]
            return valid_words[:6]
            
        except Exception as e:
            logger.error(f"Error generating words with LLM: {e}")
            return []
    
    def _entity_matches_theme(self, entity, theme_name: str) -> bool:
        """Check if an entity matches the theme."""
        theme_lower = theme_name.lower()
        entity_name_lower = entity.entity_name.lower()
        
        # Simple keyword matching - can be enhanced with semantic similarity
        if any(keyword in theme_lower for keyword in ["character", "people", "person"]):
            return entity.entity_type.value in ["CHARACTER", "ACTOR", "HISTORICAL_FIGURE"]
        elif "food" in theme_lower or "dish" in theme_lower:
            return entity.entity_type.value == "FOOD"
        elif "drink" in theme_lower or "beverage" in theme_lower:
            return entity.entity_type.value == "DRINK"
        elif "place" in theme_lower or "location" in theme_lower or "city" in theme_lower:
            return entity.entity_type.value in ["LOCATION", "LANDMARK"]
        else:
            # Generic match - use semantic similarity if available
            return True
    
    def _create_puzzle_object(self, grid_data: Dict[str, Any], constraints: GenerationConstraints) -> Puzzle:
        """Create a complete Puzzle object from grid data."""
        
        puzzle_id = f"p_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create puzzle categories
        solution = []
        for cat_data in grid_data["categories"]:
            theme = cat_data["theme"]
            words = cat_data["words"]
            
            # Map connection type to CategoryType enum
            try:
                category_type = CategoryType(theme.get("connection_type", "OTHER"))
            except ValueError:
                category_type = CategoryType.OTHER
            
            # Map tier to CulturalSpecificityTier enum  
            try:
                specificity_tier = CulturalSpecificityTier(theme.get("tier", 2))
            except ValueError:
                specificity_tier = CulturalSpecificityTier.BROADLY_BRITISH
            
            category = PuzzleCategory(
                category_name=theme["theme"],
                words=words,
                category_type=category_type,
                specificity_tier=specificity_tier
            )
            solution.append(category)
        
        # Create generation metadata
        metadata = GenerationMetadata(
            creator_model=self.model_name,
            creation_timestamp=datetime.utcnow(),
            constraints_used=constraints,
            generation_attempts=1
        )
        
        # Create puzzle
        puzzle = Puzzle(
            puzzle_id=puzzle_id,
            status=PuzzleStatus.DRAFT,
            grid=grid_data["grid"],
            solution=solution,
            generation_metadata=metadata
        )
        
        return puzzle
    
    def _format_constraints_for_prompt(self, constraints: GenerationConstraints) -> str:
        """Format constraints for inclusion in prompts."""
        if not constraints:
            return ""
        
        lines = ["Constraints:"]
        
        if constraints.cultural_specificity_tiers:
            tier_names = [f"Tier {tier.value}" for tier in constraints.cultural_specificity_tiers]
            lines.append(f"- Cultural tiers allowed: {', '.join(tier_names)}")
        
        if constraints.category_types:
            type_names = [ct.value for ct in constraints.category_types]
            lines.append(f"- Category types allowed: {', '.join(type_names)}")
        
        if constraints.required_domains:
            lines.append(f"- Required domains: {', '.join(constraints.required_domains)}")
        
        if constraints.theme:
            lines.append(f"- Theme: {constraints.theme}")
        
        return "\n".join(lines) if len(lines) > 1 else ""
    
    def _get_fallback_themes(self) -> List[List[Dict[str, Any]]]:
        """Provide fallback themes if LLM generation fails."""
        return [
            [
                {"theme": "Famous British Authors", "connection_type": "CULTURAL_LITERATURE", "tier": 1},
                {"theme": "Parts of a Full English Breakfast", "connection_type": "CULINARY", "tier": 2},
                {"theme": "Characters in Fawlty Towers", "connection_type": "CULTURAL_TV", "tier": 3},
                {"theme": "Words that can follow 'BLACK'", "connection_type": "CONCEPTUAL_FILL_IN_THE_BLANK", "tier": 4}
            ],
            [
                {"theme": "British TV Presenters", "connection_type": "CULTURAL_TV", "tier": 2},
                {"theme": "Scottish Cities", "connection_type": "GEOGRAPHICAL", "tier": 2},
                {"theme": "Types of British Cheese", "connection_type": "CULINARY", "tier": 3},
                {"theme": "Cockney Rhyming Slang", "connection_type": "LINGUISTIC_RHYME", "tier": 4}
            ]
        ]