"""Integration tests for the puzzle generation pipeline."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.pipeline import PuzzlePipeline
from app.models.puzzles import (
    Puzzle, PuzzleCategory, GenerationConstraints, GenerationMetadata,
    CategoryType, DifficultyLevel, PuzzleStatus
)
from app.models.entities import CulturalSpecificityTier
from app.database import KnowledgeGraphDB, VectorStore, CacheManager


class TestPuzzlePipeline:
    """Integration tests for the puzzle generation pipeline."""
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph."""
        kg = Mock(spec=KnowledgeGraphDB)
        kg.health_check.return_value = True
        kg.get_statistics.return_value = {"total_entities": 100}
        return kg
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        vs = Mock(spec=VectorStore)
        vs.calculate_category_cohesion.return_value = 0.8
        vs.get_statistics.return_value = {"total_entities": 100}
        vs.entities = {}
        return vs
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        cache = Mock(spec=CacheManager)
        cache.health_check.return_value = True
        cache.get_cache_stats.return_value = {"total_keys": 0}
        cache.cache_puzzle.return_value = True
        return cache
    
    @pytest.fixture
    def pipeline(self, mock_knowledge_graph, mock_vector_store, mock_cache_manager):
        """Create a pipeline with mocked dependencies."""
        return PuzzlePipeline(
            knowledge_graph=mock_knowledge_graph,
            vector_store=mock_vector_store,
            cache_manager=mock_cache_manager
        )
    
    def create_sample_puzzle(self):
        """Create a sample puzzle for testing."""
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
        
        grid = [
            ["Shakespeare", "Bacon", "Basil", "Cab"],
            ["Dickens", "Eggs", "Sybil", "Pudding"],
            ["Austen", "Sausages", "Manuel", "Belt"],
            ["Christie", "Beans", "Polly", "Sheep"]
        ]
        
        metadata = GenerationMetadata(
            creator_model="gpt-4-turbo",
            creation_timestamp=datetime.utcnow()
        )
        
        return Puzzle(
            puzzle_id="test-puzzle",
            status=PuzzleStatus.DRAFT,
            grid=grid,
            solution=categories,
            generation_metadata=metadata
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.creator_agent is not None
        assert pipeline.trickster_agent is not None
        assert pipeline.linguist_agent is not None
        assert pipeline.fact_checker_agent is not None
        assert pipeline.judge_agent is not None
        
        # Check initial stats
        assert pipeline.stats["total_puzzles_generated"] == 0
        assert pipeline.stats["successful_generations"] == 0
        assert pipeline.stats["failed_generations"] == 0
    
    def test_get_status(self, pipeline):
        """Test getting pipeline status."""
        status = pipeline.get_status()
        
        assert status["pipeline_status"] == "operational"
        assert "agents" in status
        assert "statistics" in status
        assert "configuration" in status
        
        # Check agents
        agents = status["agents"]
        assert "creator" in agents
        assert "trickster" in agents
        assert "linguist" in agents
        assert "fact_checker" in agents
        assert "judge" in agents
    
    @patch('app.agents.creator.CreatorAgent.process')
    @pytest.mark.asyncio
    async def test_creator_stage_success(self, mock_creator_process, pipeline):
        """Test successful creator stage."""
        # Mock successful creation
        sample_puzzle = self.create_sample_puzzle()
        mock_creator_process.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict()
        }
        
        result = await pipeline._run_creator_stage(None)
        
        assert result["success"] is True
        assert "puzzle" in result
        mock_creator_process.assert_called_once()
    
    @patch('app.agents.creator.CreatorAgent.process')
    @pytest.mark.asyncio
    async def test_creator_stage_failure(self, mock_creator_process, pipeline):
        """Test creator stage failure with retry."""
        # Mock failure
        mock_creator_process.return_value = {
            "success": False,
            "error": "Generation failed"
        }
        
        result = await pipeline._run_creator_stage(None)
        
        assert result["success"] is False
        assert "error" in result
        # Should retry based on settings.max_generation_attempts
        assert mock_creator_process.call_count > 1
    
    @patch('app.agents.trickster.TricksterAgent.process')
    @patch('app.agents.linguist.LinguistAgent.process')
    @patch('app.agents.fact_checker.FactCheckerAgent.process')
    @pytest.mark.asyncio
    async def test_curation_stage_success(self, mock_fact_checker, mock_linguist, mock_trickster, pipeline):
        """Test successful curation stage."""
        sample_puzzle = self.create_sample_puzzle()
        
        # Mock all agents to return success
        mock_trickster.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict()
        }
        mock_linguist.return_value = {
            "success": True, 
            "puzzle": sample_puzzle.dict()
        }
        mock_fact_checker.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict(),
            "fact_check_results": {"flagged": 0}
        }
        
        result = await pipeline._run_curation_stage(sample_puzzle)
        
        assert result["success"] is True
        mock_trickster.assert_called_once()
        mock_linguist.assert_called_once()
        mock_fact_checker.assert_called_once()
    
    @patch('app.agents.fact_checker.FactCheckerAgent.process')
    @pytest.mark.asyncio
    async def test_curation_stage_too_many_flagged(self, mock_fact_checker, pipeline):
        """Test curation stage when too many categories are flagged."""
        sample_puzzle = self.create_sample_puzzle()
        
        # Mock fact checker to flag too many categories
        mock_fact_checker.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict(),
            "fact_check_results": {"flagged": 3}  # Too many flagged
        }
        
        with patch('app.agents.trickster.TricksterAgent.process') as mock_trickster, \
             patch('app.agents.linguist.LinguistAgent.process') as mock_linguist:
            
            mock_trickster.return_value = {"success": True, "puzzle": sample_puzzle.dict()}
            mock_linguist.return_value = {"success": True, "puzzle": sample_puzzle.dict()}
            
            result = await pipeline._run_curation_stage(sample_puzzle)
            
            assert result["success"] is False
            assert "fact-checker" in result["error"]
    
    @patch('app.agents.judge.JudgeAgent.process')
    @pytest.mark.asyncio
    async def test_assessment_stage_success(self, mock_judge, pipeline):
        """Test successful assessment stage."""
        sample_puzzle = self.create_sample_puzzle()
        
        mock_judge.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict(),
            "difficulty_analysis": {"is_balanced": True}
        }
        
        result = await pipeline._run_assessment_stage(sample_puzzle)
        
        assert result["success"] is True
        assert "difficulty_analysis" in result
        mock_judge.assert_called_once()
    
    @patch('app.agents.judge.JudgeAgent.process')
    @pytest.mark.asyncio
    async def test_assessment_stage_failure(self, mock_judge, pipeline):
        """Test assessment stage failure."""
        sample_puzzle = self.create_sample_puzzle()
        
        mock_judge.return_value = {
            "success": False,
            "error": "Assessment failed"
        }
        
        result = await pipeline._run_assessment_stage(sample_puzzle)
        
        assert result["success"] is False
        assert "error" in result
    
    @patch('app.pipeline.puzzle_pipeline.PuzzlePipeline._run_creator_stage')
    @patch('app.pipeline.puzzle_pipeline.PuzzlePipeline._run_curation_stage')
    @patch('app.pipeline.puzzle_pipeline.PuzzlePipeline._run_assessment_stage')
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, mock_assessment, mock_curation, mock_creation, pipeline):
        """Test successful full pipeline execution."""
        sample_puzzle = self.create_sample_puzzle()
        
        # Mock all stages to succeed
        mock_creation.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict()
        }
        mock_curation.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict()
        }
        mock_assessment.return_value = {
            "success": True,
            "puzzle": sample_puzzle.dict(),
            "difficulty_analysis": {"is_balanced": True}
        }
        
        result = await pipeline.generate_puzzle()
        
        assert result["success"] is True
        assert "puzzle" in result
        assert "processing_time_seconds" in result
        assert result["pipeline_stages_completed"] == 3
        
        # Check that stats were updated
        assert pipeline.stats["total_puzzles_generated"] == 1
        assert pipeline.stats["successful_generations"] == 1
        assert pipeline.stats["failed_generations"] == 0
    
    @patch('app.pipeline.puzzle_pipeline.PuzzlePipeline._run_creator_stage')
    @pytest.mark.asyncio
    async def test_pipeline_creation_failure(self, mock_creation, pipeline):
        """Test pipeline failure at creation stage."""
        mock_creation.return_value = {
            "success": False,
            "error": "Creation failed"
        }
        
        result = await pipeline.generate_puzzle()
        
        assert result["success"] is False
        assert "error" in result
        
        # Check that stats were updated
        assert pipeline.stats["total_puzzles_generated"] == 1
        assert pipeline.stats["successful_generations"] == 0
        assert pipeline.stats["failed_generations"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_puzzle(self, pipeline):
        """Test puzzle validation."""
        # Valid puzzle
        valid_puzzle = self.create_sample_puzzle()
        valid_puzzle.status = PuzzleStatus.ASSESSED
        
        validation_result = await pipeline.validate_puzzle(valid_puzzle)
        
        assert validation_result["is_valid"] is True
        assert validation_result["puzzle_id"] == "test-puzzle"
        assert len(validation_result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_invalid_puzzle(self, pipeline):
        """Test validation of invalid puzzle."""
        # Create invalid puzzle (wrong grid size)
        invalid_puzzle = self.create_sample_puzzle()
        invalid_puzzle.grid = [["A", "B"]]  # Wrong size
        
        validation_result = await pipeline.validate_puzzle(invalid_puzzle)
        
        assert validation_result["is_valid"] is False
        assert len(validation_result["issues"]) > 0
    
    def test_get_pipeline_metrics(self, pipeline, mock_knowledge_graph, mock_cache_manager, mock_vector_store):
        """Test getting pipeline metrics."""
        metrics = pipeline.get_pipeline_metrics()
        
        assert "generation_metrics" in metrics
        assert "component_health" in metrics
        assert "cache_stats" in metrics
        assert "knowledge_graph_stats" in metrics
        assert "vector_store_stats" in metrics
        
        # Verify mocks were called
        mock_knowledge_graph.health_check.assert_called_once()
        mock_cache_manager.health_check.assert_called_once()
        mock_knowledge_graph.get_statistics.assert_called_once()
        mock_cache_manager.get_cache_stats.assert_called_once()
        mock_vector_store.get_statistics.assert_called_once()
    
    def test_update_stats(self, pipeline):
        """Test stats updating."""
        initial_total = pipeline.stats["total_puzzles_generated"]
        initial_successful = pipeline.stats["successful_generations"]
        
        pipeline._update_stats(success=True, processing_time=30.5)
        
        assert pipeline.stats["total_puzzles_generated"] == initial_total + 1
        assert pipeline.stats["successful_generations"] == initial_successful + 1
        assert pipeline.stats["average_processing_time"] == 30.5
        assert pipeline.stats["last_generation_time"] is not None
    
    @pytest.mark.asyncio
    async def test_cache_puzzle(self, pipeline, mock_cache_manager):
        """Test puzzle caching."""
        sample_puzzle = self.create_sample_puzzle()
        
        await pipeline._cache_puzzle(sample_puzzle)
        
        mock_cache_manager.cache_puzzle.assert_called_once_with(
            sample_puzzle.puzzle_id,
            sample_puzzle.dict(),
            ttl=3600  # Default from settings
        )