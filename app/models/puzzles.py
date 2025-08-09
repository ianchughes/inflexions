"""Puzzle data models for the AI Editorial Engine."""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .entities import CulturalSpecificityTier


class PuzzleStatus(str, Enum):
    """Status of a puzzle in the editorial pipeline."""
    DRAFT = "DRAFT"
    CURATED = "CURATED" 
    ASSESSED = "ASSESSED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"


class DifficultyLevel(str, Enum):
    """Difficulty levels for puzzle categories."""
    YELLOW = "YELLOW"  # Easiest
    GREEN = "GREEN"    # Easy-Medium
    BLUE = "BLUE"      # Medium-Hard
    PURPLE = "PURPLE"  # Hardest


class CategoryType(str, Enum):
    """Types of puzzle categories based on connection logic."""
    SEMANTIC_HYPONYM = "SEMANTIC_HYPONYM"  # e.g., "Types of dogs"
    SEMANTIC_MERONYM = "SEMANTIC_MERONYM"  # e.g., "Parts of a car"
    SEMANTIC_SYNONYM = "SEMANTIC_SYNONYM"  # e.g., "Words meaning happy"
    CULTURAL_TV = "CULTURAL_TV"            # e.g., "Characters in Fawlty Towers"
    CULTURAL_FILM = "CULTURAL_FILM"        # e.g., "James Bond actors"
    CULTURAL_MUSIC = "CULTURAL_MUSIC"      # e.g., "Beatles albums"
    CULTURAL_LITERATURE = "CULTURAL_LITERATURE"  # e.g., "Dickens novels"
    CULTURAL_HISTORY = "CULTURAL_HISTORY"  # e.g., "Tudor monarchs"
    GEOGRAPHICAL = "GEOGRAPHICAL"          # e.g., "Scottish cities"
    CULINARY = "CULINARY"                 # e.g., "Ingredients in haggis"
    LINGUISTIC_RHYME = "LINGUISTIC_RHYME"  # e.g., "Words that rhyme with 'mate'"
    LINGUISTIC_HOMOPHONE = "LINGUISTIC_HOMOPHONE"  # e.g., "Homophones"
    CONCEPTUAL_WORDPLAY = "CONCEPTUAL_WORDPLAY"    # e.g., "Words ending in -ough"
    CONCEPTUAL_FILL_IN_THE_BLANK = "CONCEPTUAL_FILL_IN_THE_BLANK"  # e.g., "Words that can follow 'black'"
    STRUCTURAL_ANAGRAM = "STRUCTURAL_ANAGRAM"      # e.g., "Anagrams of each other"
    STRUCTURAL_LENGTH = "STRUCTURAL_LENGTH"        # e.g., "Four-letter words"
    OTHER = "OTHER"


class DifficultyMetrics(BaseModel):
    """Quantitative metrics for assessing category difficulty."""
    
    cosine_similarity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Average pairwise cosine similarity between word embeddings"
    )
    
    word_frequency: str = Field(
        ..., 
        description="Average word frequency rating (high/medium/low)"
    )
    
    polysemy_score: Optional[float] = Field(
        None,
        description="Average number of meanings per word"
    )
    
    cultural_specificity_variance: Optional[float] = Field(
        None,
        description="Variance in cultural specificity among words"
    )


class PuzzleCategory(BaseModel):
    """A single category within a puzzle."""
    
    category_name: str = Field(..., description="Name of the category")
    words: List[str] = Field(..., min_items=4, max_items=4, description="Four words in this category")
    category_type: CategoryType = Field(..., description="Type of connection logic")
    specificity_tier: CulturalSpecificityTier = Field(..., description="Cultural specificity tier")
    difficulty: Optional[DifficultyLevel] = Field(None, description="Assigned difficulty level")
    justification: Optional[str] = Field(None, description="Explanation for difficulty assignment")
    
    # Metrics for difficulty assessment
    difficulty_metrics: Optional[DifficultyMetrics] = Field(
        None,
        description="Quantitative metrics for difficulty assessment"
    )
    
    # Editorial agent processing flags
    processed_by_trickster: bool = Field(default=False, description="Whether Trickster agent has processed")
    processed_by_linguist: bool = Field(default=False, description="Whether Linguist agent has processed")
    processed_by_fact_checker: bool = Field(default=False, description="Whether Fact-Checker agent has processed")
    
    # Validation data
    fact_check_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Results from fact-checking validation"
    )
    
    class Config:
        use_enum_values = True


class GenerationConstraints(BaseModel):
    """Constraints for puzzle generation."""
    
    target_difficulty_distribution: Optional[Dict[DifficultyLevel, int]] = Field(
        None,
        description="Desired number of categories per difficulty level"
    )
    
    cultural_specificity_tiers: Optional[List[CulturalSpecificityTier]] = Field(
        None,
        description="Allowed cultural specificity tiers"
    )
    
    category_types: Optional[List[CategoryType]] = Field(
        None,
        description="Allowed category types"
    )
    
    excluded_entities: Optional[List[str]] = Field(
        None,
        description="Entity IDs to exclude from generation"
    )
    
    required_domains: Optional[List[str]] = Field(
        None,
        description="Required cultural domains (e.g., 'TV', 'Music', 'Food')"
    )
    
    theme: Optional[str] = Field(
        None,
        description="Optional thematic constraint for the puzzle"
    )
    
    class Config:
        use_enum_values = True


class GenerationMetadata(BaseModel):
    """Metadata about puzzle generation process."""
    
    creator_model: str = Field(..., description="LLM model used for generation")
    creation_timestamp: datetime = Field(..., description="When the puzzle was generated")
    
    difficulty_metrics: Dict[str, DifficultyMetrics] = Field(
        default_factory=dict,
        description="Difficulty metrics for each category"
    )
    
    generation_attempts: int = Field(default=1, description="Number of generation attempts")
    constraints_used: Optional[GenerationConstraints] = Field(None, description="Constraints applied during generation")
    
    # Agent processing metadata
    trickster_metadata: Optional[Dict[str, Any]] = Field(None, description="Trickster agent processing data")
    linguist_metadata: Optional[Dict[str, Any]] = Field(None, description="Linguist agent processing data")
    fact_checker_metadata: Optional[Dict[str, Any]] = Field(None, description="Fact-checker agent processing data")
    judge_metadata: Optional[Dict[str, Any]] = Field(None, description="Judge agent assessment data")
    
    # Performance metrics
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Puzzle(BaseModel):
    """Complete puzzle object as defined in the specification."""
    
    puzzle_id: str = Field(..., description="Unique identifier for the puzzle")
    status: PuzzleStatus = Field(default=PuzzleStatus.DRAFT, description="Current status in pipeline")
    
    # 4x4 grid of 16 words
    grid: List[List[str]] = Field(
        ..., 
        min_items=4, 
        max_items=4,
        description="4x4 grid of words"
    )
    
    # Solution with 4 categories
    solution: List[PuzzleCategory] = Field(
        ..., 
        min_items=4, 
        max_items=4,
        description="Four puzzle categories with solutions"
    )
    
    # Generation and processing metadata
    generation_metadata: GenerationMetadata = Field(..., description="Metadata about generation process")
    
    # Human review
    human_review_status: Optional[str] = Field(None, description="Status of human review")
    human_review_notes: Optional[str] = Field(None, description="Notes from human reviewer")
    
    # Publication data
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    play_statistics: Optional[Dict[str, Any]] = Field(None, description="Player engagement metrics")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def validate_grid(self) -> bool:
        """Validate that grid contains exactly 16 unique words from solution categories."""
        grid_words = [word for row in self.grid for word in row]
        solution_words = [word for category in self.solution for word in category.words]
        
        return (
            len(grid_words) == 16 and
            len(set(grid_words)) == 16 and
            set(grid_words) == set(solution_words)
        )
    
    def get_category_by_difficulty(self, difficulty: DifficultyLevel) -> Optional[PuzzleCategory]:
        """Get category by difficulty level."""
        for category in self.solution:
            if category.difficulty == difficulty:
                return category
        return None
    
    def get_words_by_difficulty_order(self) -> List[List[str]]:
        """Get words grouped by difficulty order (Yellow, Green, Blue, Purple)."""
        difficulty_order = [DifficultyLevel.YELLOW, DifficultyLevel.GREEN, 
                          DifficultyLevel.BLUE, DifficultyLevel.PURPLE]
        
        ordered_groups = []
        for difficulty in difficulty_order:
            category = self.get_category_by_difficulty(difficulty)
            if category:
                ordered_groups.append(category.words)
        
        return ordered_groups


class PuzzleGenerationRequest(BaseModel):
    """Request model for puzzle generation endpoint."""
    
    constraints: Optional[GenerationConstraints] = Field(
        None,
        description="Optional constraints for puzzle generation"
    )
    
    class Config:
        use_enum_values = True


class PuzzleGenerationResponse(BaseModel):
    """Response model for puzzle generation endpoint."""
    
    puzzle: Puzzle = Field(..., description="Generated puzzle")
    success: bool = Field(..., description="Whether generation was successful")
    message: Optional[str] = Field(None, description="Additional message or error details")
    
    class Config:
        use_enum_values = True