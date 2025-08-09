"""Knowledge Graph Entity models for UK Cultural data."""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities in the UK Cultural Knowledge Graph."""
    TV_SHOW = "TV_SHOW"
    MOVIE = "MOVIE"
    BOOK = "BOOK"
    AUTHOR = "AUTHOR"
    ACTOR = "ACTOR"
    CHARACTER = "CHARACTER"
    HISTORICAL_FIGURE = "HISTORICAL_FIGURE"
    HISTORICAL_EVENT = "HISTORICAL_EVENT"
    LOCATION = "LOCATION"
    LANDMARK = "LANDMARK"
    FOOD = "FOOD"
    DRINK = "DRINK"
    BRAND = "BRAND"
    MUSIC_ARTIST = "MUSIC_ARTIST"
    SONG = "SONG"
    ALBUM = "ALBUM"
    SLANG_TERM = "SLANG_TERM"
    PHRASE = "PHRASE"
    INSTITUTION = "INSTITUTION"
    SPORT = "SPORT"
    TRADITION = "TRADITION"
    OTHER = "OTHER"


class CulturalSpecificityTier(int, Enum):
    """Cultural specificity tiers for controlling puzzle accessibility."""
    GLOBAL_PAN_UK = 1  # Universally known (e.g., The Beatles, Shakespeare)
    BROADLY_BRITISH = 2  # Household names in the UK (e.g., Only Fools and Horses, Marmite)
    CULTURALLY_ATTUNED = 3  # Requires deeper cultural fluency (e.g., The Thick of It characters)
    NICHE_REGIONAL = 4  # Highly specific knowledge (e.g., Glaswegian slang, regional names)


class KnowledgeGraphEntity(BaseModel):
    """A single entity in the UK Cultural Knowledge Graph."""
    
    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_name: str = Field(..., description="Primary name of the entity")
    entity_type: EntityType = Field(..., description="Type classification of the entity")
    
    attributes: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Type-specific attributes (genre, creator, etc.)"
    )
    
    cultural_specificity_tier: CulturalSpecificityTier = Field(
        ..., 
        description="Tier indicating cultural specificity for puzzle difficulty"
    )
    
    related_entities: List[str] = Field(
        default_factory=list,
        description="List of related entity IDs"
    )
    
    source: Optional[str] = Field(
        None, 
        description="Source URL or reference for the entity information"
    )
    
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names or spellings for the entity"
    )
    
    description: Optional[str] = Field(
        None,
        description="Brief description of the entity"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Additional tags for categorization and search"
    )
    
    embedding: Optional[List[float]] = Field(
        None,
        description="Vector embedding for semantic similarity"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when entity was created"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when entity was last updated"
    )
    
    validation_status: str = Field(
        default="pending",
        description="Status of fact-checking validation"
    )
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EntityRelationship(BaseModel):
    """Represents a relationship between two entities in the knowledge graph."""
    
    source_entity_id: str = Field(..., description="ID of the source entity")
    target_entity_id: str = Field(..., description="ID of the target entity")
    relationship_type: str = Field(..., description="Type of relationship (e.g., 'created_by', 'appears_in')")
    strength: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0,
        description="Strength of the relationship (0.0 to 1.0)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the relationship"
    )
    
    class Config:
        use_enum_values = True


class EntityQuery(BaseModel):
    """Query parameters for searching entities in the knowledge graph."""
    
    entity_types: Optional[List[EntityType]] = Field(
        None,
        description="Filter by entity types"
    )
    
    cultural_specificity_tiers: Optional[List[CulturalSpecificityTier]] = Field(
        None,
        description="Filter by cultural specificity tiers"
    )
    
    search_text: Optional[str] = Field(
        None,
        description="Text to search in entity names and descriptions"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="Filter by tags"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip"
    )
    
    semantic_search: Optional[str] = Field(
        None,
        description="Text for semantic similarity search using embeddings"
    )
    
    class Config:
        use_enum_values = True