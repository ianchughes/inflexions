"""Tests for data models."""

import pytest
from datetime import datetime
from app.models.entities import (
    KnowledgeGraphEntity, EntityType, CulturalSpecificityTier, EntityQuery
)
from app.models.puzzles import (
    Puzzle, PuzzleCategory, GenerationConstraints, GenerationMetadata,
    CategoryType, DifficultyLevel, PuzzleStatus, DifficultyMetrics
)


class TestKnowledgeGraphEntity:
    """Tests for KnowledgeGraphEntity model."""
    
    def test_create_basic_entity(self):
        """Test creating a basic entity."""
        entity = KnowledgeGraphEntity(
            entity_id="test-entity",
            entity_name="Test Entity",
            entity_type=EntityType.TV_SHOW,
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH
        )
        
        assert entity.entity_id == "test-entity"
        assert entity.entity_name == "Test Entity"
        assert entity.entity_type == EntityType.TV_SHOW
        assert entity.cultural_specificity_tier == CulturalSpecificityTier.BROADLY_BRITISH
        assert entity.attributes == {}
        assert entity.related_entities == []
        assert entity.validation_status == "pending"
    
    def test_entity_with_attributes(self):
        """Test creating entity with custom attributes."""
        entity = KnowledgeGraphEntity(
            entity_id="tv-show-test",
            entity_name="Test Show",
            entity_type=EntityType.TV_SHOW,
            cultural_specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            attributes={
                "genre": "Comedy",
                "years": "2000-2010",
                "characters": ["Alice", "Bob"]
            },
            tags=["comedy", "british"],
            description="A test TV show"
        )
        
        assert entity.attributes["genre"] == "Comedy"
        assert entity.attributes["characters"] == ["Alice", "Bob"]
        assert entity.tags == ["comedy", "british"]
        assert entity.description == "A test TV show"
    
    def test_entity_enum_values(self):
        """Test that enum values are properly handled."""
        entity = KnowledgeGraphEntity(
            entity_id="test",
            entity_name="Test",
            entity_type=EntityType.AUTHOR,
            cultural_specificity_tier=CulturalSpecificityTier.NICHE_REGIONAL
        )
        
        # Test JSON serialization preserves enum values
        entity_dict = entity.dict()
        assert entity_dict["entity_type"] == "AUTHOR"
        assert entity_dict["cultural_specificity_tier"] == 4


class TestPuzzleModels:
    """Tests for puzzle-related models."""
    
    def test_difficulty_metrics(self):
        """Test DifficultyMetrics model."""
        metrics = DifficultyMetrics(
            cosine_similarity=0.85,
            word_frequency="high",
            polysemy_score=2.3,
            cultural_specificity_variance=0.1
        )
        
        assert metrics.cosine_similarity == 0.85
        assert metrics.word_frequency == "high"
        assert metrics.polysemy_score == 2.3
        assert metrics.cultural_specificity_variance == 0.1
    
    def test_puzzle_category(self):
        """Test PuzzleCategory model."""
        category = PuzzleCategory(
            category_name="British Authors",
            words=["Shakespeare", "Dickens", "Austen", "Christie"],
            category_type=CategoryType.CULTURAL_LITERATURE,
            specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            difficulty=DifficultyLevel.YELLOW,
            justification="Well-known global authors"
        )
        
        assert category.category_name == "British Authors"
        assert len(category.words) == 4
        assert category.category_type == CategoryType.CULTURAL_LITERATURE
        assert category.difficulty == DifficultyLevel.YELLOW
        assert not category.processed_by_trickster
        assert not category.processed_by_linguist
        assert not category.processed_by_fact_checker
    
    def test_generation_metadata(self):
        """Test GenerationMetadata model."""
        metadata = GenerationMetadata(
            creator_model="gpt-4-turbo",
            creation_timestamp=datetime.utcnow(),
            generation_attempts=2,
            processing_time_seconds=45.5
        )
        
        assert metadata.creator_model == "gpt-4-turbo"
        assert metadata.generation_attempts == 2
        assert metadata.processing_time_seconds == 45.5
        assert isinstance(metadata.creation_timestamp, datetime)
    
    def test_complete_puzzle(self):
        """Test complete Puzzle model."""
        # Create categories
        categories = [
            PuzzleCategory(
                category_name="British Authors",
                words=["Shakespeare", "Dickens", "Austen", "Christie"],
                category_type=CategoryType.CULTURAL_LITERATURE,
                specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
                difficulty=DifficultyLevel.YELLOW
            ),
            PuzzleCategory(
                category_name="Full English Breakfast",
                words=["Bacon", "Eggs", "Sausages", "Beans"],
                category_type=CategoryType.CULINARY,
                specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
                difficulty=DifficultyLevel.GREEN
            ),
            PuzzleCategory(
                category_name="Fawlty Towers Characters",
                words=["Basil", "Sybil", "Manuel", "Polly"],
                category_type=CategoryType.CULTURAL_TV,
                specificity_tier=CulturalSpecificityTier.CULTURALLY_ATTUNED,
                difficulty=DifficultyLevel.BLUE
            ),
            PuzzleCategory(
                category_name="Words after BLACK",
                words=["Cab", "Pudding", "Belt", "Sheep"],
                category_type=CategoryType.CONCEPTUAL_FILL_IN_THE_BLANK,
                specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
                difficulty=DifficultyLevel.PURPLE
            )
        ]
        
        # Create grid (shuffled words)
        grid = [
            ["Shakespeare", "Bacon", "Basil", "Cab"],
            ["Dickens", "Eggs", "Sybil", "Pudding"], 
            ["Austen", "Sausages", "Manuel", "Belt"],
            ["Christie", "Beans", "Polly", "Sheep"]
        ]
        
        # Create metadata
        metadata = GenerationMetadata(
            creator_model="gpt-4-turbo",
            creation_timestamp=datetime.utcnow()
        )
        
        # Create puzzle
        puzzle = Puzzle(
            puzzle_id="test-puzzle-001",
            status=PuzzleStatus.ASSESSED,
            grid=grid,
            solution=categories,
            generation_metadata=metadata
        )
        
        assert puzzle.puzzle_id == "test-puzzle-001"
        assert puzzle.status == PuzzleStatus.ASSESSED
        assert len(puzzle.grid) == 4
        assert len(puzzle.solution) == 4
        assert puzzle.validate_grid()
    
    def test_puzzle_validation(self):
        """Test puzzle grid validation."""
        # Valid puzzle
        categories = [
            PuzzleCategory(
                category_name="Test Category",
                words=["A", "B", "C", "D"],
                category_type=CategoryType.OTHER,
                specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH
            )
        ]
        
        grid = [["A", "B", "C", "D"]]
        metadata = GenerationMetadata(
            creator_model="test",
            creation_timestamp=datetime.utcnow()
        )
        
        # This should fail validation (not 4x4 grid, only 1 category)
        puzzle = Puzzle(
            puzzle_id="test",
            grid=grid,
            solution=categories,
            generation_metadata=metadata
        )
        
        assert not puzzle.validate_grid()
    
    def test_difficulty_distribution_methods(self):
        """Test puzzle difficulty distribution methods."""
        categories = [
            PuzzleCategory(
                category_name="Yellow Category",
                words=["A", "B", "C", "D"],
                category_type=CategoryType.OTHER,
                specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
                difficulty=DifficultyLevel.YELLOW
            ),
            PuzzleCategory(
                category_name="Green Category", 
                words=["E", "F", "G", "H"],
                category_type=CategoryType.OTHER,
                specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
                difficulty=DifficultyLevel.GREEN
            ),
            PuzzleCategory(
                category_name="Blue Category",
                words=["I", "J", "K", "L"],
                category_type=CategoryType.OTHER,
                specificity_tier=CulturalSpecificityTier.CULTURALLY_ATTUNED,
                difficulty=DifficultyLevel.BLUE
            ),
            PuzzleCategory(
                category_name="Purple Category",
                words=["M", "N", "O", "P"],
                category_type=CategoryType.OTHER,
                specificity_tier=CulturalSpecificityTier.NICHE_REGIONAL,
                difficulty=DifficultyLevel.PURPLE
            )
        ]
        
        grid = [
            ["A", "E", "I", "M"],
            ["B", "F", "J", "N"],
            ["C", "G", "K", "O"],
            ["D", "H", "L", "P"]
        ]
        
        metadata = GenerationMetadata(
            creator_model="test",
            creation_timestamp=datetime.utcnow()
        )
        
        puzzle = Puzzle(
            puzzle_id="test-difficulty",
            grid=grid,
            solution=categories,
            generation_metadata=metadata
        )
        
        # Test get_category_by_difficulty
        yellow_cat = puzzle.get_category_by_difficulty(DifficultyLevel.YELLOW)
        assert yellow_cat is not None
        assert yellow_cat.category_name == "Yellow Category"
        
        # Test get_words_by_difficulty_order
        ordered_words = puzzle.get_words_by_difficulty_order()
        assert len(ordered_words) == 4
        assert ordered_words[0] == ["A", "B", "C", "D"]  # Yellow
        assert ordered_words[1] == ["E", "F", "G", "H"]  # Green
        assert ordered_words[2] == ["I", "J", "K", "L"]  # Blue
        assert ordered_words[3] == ["M", "N", "O", "P"]  # Purple


class TestEntityQuery:
    """Tests for EntityQuery model."""
    
    def test_basic_query(self):
        """Test basic entity query."""
        query = EntityQuery(
            entity_types=[EntityType.TV_SHOW, EntityType.MOVIE],
            search_text="comedy",
            limit=20,
            offset=10
        )
        
        assert len(query.entity_types) == 2
        assert EntityType.TV_SHOW in query.entity_types
        assert query.search_text == "comedy"
        assert query.limit == 20
        assert query.offset == 10
    
    def test_query_with_all_filters(self):
        """Test entity query with all possible filters."""
        query = EntityQuery(
            entity_types=[EntityType.AUTHOR],
            cultural_specificity_tiers=[CulturalSpecificityTier.GLOBAL_PAN_UK],
            search_text="literature",
            tags=["classic", "british"],
            limit=5,
            offset=0,
            semantic_search="famous writers"
        )
        
        assert query.entity_types == [EntityType.AUTHOR]
        assert query.cultural_specificity_tiers == [CulturalSpecificityTier.GLOBAL_PAN_UK]
        assert query.search_text == "literature"
        assert query.tags == ["classic", "british"]
        assert query.semantic_search == "famous writers"


class TestGenerationConstraints:
    """Tests for GenerationConstraints model."""
    
    def test_basic_constraints(self):
        """Test basic generation constraints."""
        constraints = GenerationConstraints(
            cultural_specificity_tiers=[
                CulturalSpecificityTier.GLOBAL_PAN_UK,
                CulturalSpecificityTier.BROADLY_BRITISH
            ],
            category_types=[CategoryType.CULTURAL_TV, CategoryType.CULINARY],
            theme="British Comedy"
        )
        
        assert len(constraints.cultural_specificity_tiers) == 2
        assert len(constraints.category_types) == 2
        assert constraints.theme == "British Comedy"
    
    def test_difficulty_distribution_constraint(self):
        """Test difficulty distribution constraints."""
        constraints = GenerationConstraints(
            target_difficulty_distribution={
                DifficultyLevel.YELLOW: 1,
                DifficultyLevel.GREEN: 1,
                DifficultyLevel.BLUE: 1,
                DifficultyLevel.PURPLE: 1
            }
        )
        
        assert len(constraints.target_difficulty_distribution) == 4
        assert constraints.target_difficulty_distribution[DifficultyLevel.YELLOW] == 1