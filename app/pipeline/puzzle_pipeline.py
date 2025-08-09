"""Main puzzle generation pipeline orchestrating all agents."""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import time

from ..models.puzzles import Puzzle, GenerationConstraints, PuzzleStatus
from ..database import KnowledgeGraphDB, VectorStore, CacheManager
from ..agents import CreatorAgent, TricksterAgent, LinguistAgent, FactCheckerAgent, JudgeAgent
from ..config import settings

logger = logging.getLogger(__name__)


class PuzzlePipeline:
    """Main pipeline that orchestrates the puzzle generation process."""
    
    def __init__(self, knowledge_graph: KnowledgeGraphDB, vector_store: VectorStore, cache_manager: CacheManager):
        """Initialize the puzzle pipeline."""
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.cache_manager = cache_manager
        
        # Initialize agents
        self.creator_agent = CreatorAgent(
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            cache_manager=cache_manager
        )
        
        self.trickster_agent = TricksterAgent(
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            cache_manager=cache_manager
        )
        
        self.linguist_agent = LinguistAgent(cache_manager=cache_manager)
        
        self.fact_checker_agent = FactCheckerAgent(
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            cache_manager=cache_manager
        )
        
        self.judge_agent = JudgeAgent(
            vector_store=vector_store,
            cache_manager=cache_manager
        )
        
        # Pipeline statistics
        self.stats = {
            "total_puzzles_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_processing_time": 0.0,
            "last_generation_time": None
        }
    
    async def generate_puzzle(self, constraints: Optional[GenerationConstraints] = None) -> Dict[str, Any]:
        """Generate a complete puzzle through the full pipeline."""
        start_time = time.time()
        
        try:
            logger.info("Starting puzzle generation pipeline")
            
            # Stage 1: Creation
            creation_result = await self._run_creator_stage(constraints)
            if not creation_result["success"]:
                return creation_result
            
            puzzle = Puzzle(**creation_result["puzzle"])
            puzzle.status = PuzzleStatus.DRAFT
            
            # Stage 2: Editorial Curation
            curation_result = await self._run_curation_stage(puzzle)
            if not curation_result["success"]:
                return curation_result
            
            puzzle = Puzzle(**curation_result["puzzle"])
            puzzle.status = PuzzleStatus.CURATED
            
            # Stage 3: Difficulty Assessment
            assessment_result = await self._run_assessment_stage(puzzle)
            if not assessment_result["success"]:
                return assessment_result
            
            puzzle = Puzzle(**assessment_result["puzzle"])
            puzzle.status = PuzzleStatus.ASSESSED
            
            # Cache the final puzzle
            await self._cache_puzzle(puzzle)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(success=True, processing_time=processing_time)
            
            # Update puzzle metadata
            puzzle.generation_metadata.processing_time_seconds = processing_time
            
            logger.info(f"Puzzle generation completed successfully in {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "puzzle": puzzle.dict(),
                "processing_time_seconds": processing_time,
                "pipeline_stages_completed": 3
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(success=False, processing_time=processing_time)
            
            logger.error(f"Pipeline failed after {processing_time:.2f} seconds: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_seconds": processing_time
            }
    
    async def _run_creator_stage(self, constraints: Optional[GenerationConstraints]) -> Dict[str, Any]:
        """Run the puzzle creation stage."""
        logger.info("Running Creator Agent stage")
        
        try:
            # Prepare input for creator agent
            input_data = {
                "constraints": constraints.dict() if constraints else {}
            }
            
            # Generate puzzle with retry logic
            max_attempts = settings.max_generation_attempts
            for attempt in range(max_attempts):
                logger.info(f"Creation attempt {attempt + 1}/{max_attempts}")
                
                result = self.creator_agent.process(input_data)
                
                if result["success"]:
                    logger.info("Creator Agent completed successfully")
                    return result
                else:
                    logger.warning(f"Creation attempt {attempt + 1} failed: {result.get('error')}")
                    
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(1)  # Brief delay before retry
            
            return {
                "success": False,
                "error": f"Creator Agent failed after {max_attempts} attempts"
            }
            
        except Exception as e:
            logger.error(f"Creator stage failed: {e}")
            return {
                "success": False,
                "error": f"Creator stage error: {str(e)}"
            }
    
    async def _run_curation_stage(self, puzzle: Puzzle) -> Dict[str, Any]:
        """Run the editorial curation stage with all editorial agents."""
        logger.info("Running Editorial Curation stage")
        
        try:
            current_puzzle = puzzle
            
            # Run Trickster Agent
            logger.info("Running Trickster Agent")
            trickster_result = self.trickster_agent.process({"puzzle": current_puzzle.dict()})
            
            if not trickster_result["success"]:
                logger.warning(f"Trickster Agent failed: {trickster_result.get('error')}")
                # Continue with original puzzle
            else:
                current_puzzle = Puzzle(**trickster_result["puzzle"])
                logger.info("Trickster Agent completed successfully")
            
            # Run Linguist Agent
            logger.info("Running Linguist Agent")
            linguist_result = self.linguist_agent.process({"puzzle": current_puzzle.dict()})
            
            if not linguist_result["success"]:
                logger.warning(f"Linguist Agent failed: {linguist_result.get('error')}")
                # Continue with current puzzle
            else:
                current_puzzle = Puzzle(**linguist_result["puzzle"])
                logger.info("Linguist Agent completed successfully")
            
            # Run Fact-Checker Agent
            logger.info("Running Fact-Checker Agent")
            fact_check_result = self.fact_checker_agent.process({"puzzle": current_puzzle.dict()})
            
            if not fact_check_result["success"]:
                logger.warning(f"Fact-Checker Agent failed: {fact_check_result.get('error')}")
                # Continue with current puzzle
            else:
                current_puzzle = Puzzle(**fact_check_result["puzzle"])
                logger.info("Fact-Checker Agent completed successfully")
                
                # Check if any categories were flagged as problematic
                fact_check_summary = fact_check_result.get("fact_check_results", {})
                flagged_count = fact_check_summary.get("flagged", 0)
                
                if flagged_count > 0:
                    logger.warning(f"{flagged_count} categories flagged by fact-checker")
                    
                    # If too many categories are flagged, reject the puzzle
                    if flagged_count > 1:  # Allow 1 flagged category but not more
                        return {
                            "success": False,
                            "error": f"Too many categories flagged by fact-checker: {flagged_count}"
                        }
            
            logger.info("Editorial Curation stage completed successfully")
            
            return {
                "success": True,
                "puzzle": current_puzzle.dict()
            }
            
        except Exception as e:
            logger.error(f"Curation stage failed: {e}")
            return {
                "success": False,
                "error": f"Curation stage error: {str(e)}"
            }
    
    async def _run_assessment_stage(self, puzzle: Puzzle) -> Dict[str, Any]:
        """Run the difficulty assessment stage."""
        logger.info("Running Judge Agent stage")
        
        try:
            # Run Judge Agent
            judge_result = self.judge_agent.process({"puzzle": puzzle.dict()})
            
            if judge_result["success"]:
                assessed_puzzle = Puzzle(**judge_result["puzzle"])
                
                # Validate difficulty distribution
                difficulty_analysis = judge_result.get("difficulty_analysis", {})
                is_balanced = difficulty_analysis.get("is_balanced", False)
                
                if not is_balanced:
                    logger.warning("Difficulty distribution is not balanced, but proceeding")
                
                logger.info("Judge Agent completed successfully")
                
                return {
                    "success": True,
                    "puzzle": assessed_puzzle.dict(),
                    "difficulty_analysis": difficulty_analysis
                }
            else:
                return {
                    "success": False,
                    "error": f"Judge Agent failed: {judge_result.get('error')}"
                }
                
        except Exception as e:
            logger.error(f"Assessment stage failed: {e}")
            return {
                "success": False,
                "error": f"Assessment stage error: {str(e)}"
            }
    
    async def _cache_puzzle(self, puzzle: Puzzle) -> None:
        """Cache the completed puzzle."""
        try:
            puzzle_data = puzzle.dict()
            success = self.cache_manager.cache_puzzle(
                puzzle.puzzle_id,
                puzzle_data,
                ttl=settings.cache_ttl_seconds
            )
            
            if success:
                logger.info(f"Puzzle {puzzle.puzzle_id} cached successfully")
            else:
                logger.warning(f"Failed to cache puzzle {puzzle.puzzle_id}")
                
        except Exception as e:
            logger.error(f"Error caching puzzle: {e}")
    
    def _update_stats(self, success: bool, processing_time: float) -> None:
        """Update pipeline statistics."""
        self.stats["total_puzzles_generated"] += 1
        self.stats["last_generation_time"] = datetime.utcnow().isoformat()
        
        if success:
            self.stats["successful_generations"] += 1
        else:
            self.stats["failed_generations"] += 1
        
        # Update average processing time
        total_successful = self.stats["successful_generations"]
        if total_successful > 0:
            current_avg = self.stats["average_processing_time"]
            new_avg = ((current_avg * (total_successful - 1)) + processing_time) / total_successful
            self.stats["average_processing_time"] = new_avg
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        return {
            "pipeline_status": "operational",
            "agents": {
                "creator": {"status": "ready", "model": self.creator_agent.model_name},
                "trickster": {"status": "ready", "model": self.trickster_agent.model_name},
                "linguist": {"status": "ready", "model": self.linguist_agent.model_name},
                "fact_checker": {"status": "ready", "model": self.fact_checker_agent.model_name},
                "judge": {"status": "ready", "model": self.judge_agent.model_name}
            },
            "statistics": self.stats,
            "configuration": {
                "max_generation_attempts": settings.max_generation_attempts,
                "enable_human_review": settings.enable_human_review,
                "cache_ttl_seconds": settings.cache_ttl_seconds
            }
        }
    
    async def validate_puzzle(self, puzzle: Puzzle) -> Dict[str, Any]:
        """Validate a puzzle for correctness and quality."""
        validation_results = {
            "puzzle_id": puzzle.puzzle_id,
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            # Validate grid structure
            if not puzzle.validate_grid():
                validation_results["is_valid"] = False
                validation_results["issues"].append("Grid validation failed: words don't match solution")
            
            # Validate categories
            if len(puzzle.solution) != 4:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Expected 4 categories, found {len(puzzle.solution)}")
            
            # Validate each category has 4 words
            for i, category in enumerate(puzzle.solution):
                if len(category.words) != 4:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Category {i+1} has {len(category.words)} words, expected 4")
            
            # Check for word uniqueness
            all_words = [word for category in puzzle.solution for word in category.words]
            if len(set(all_words)) != 16:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Duplicate words found in puzzle")
            
            # Validate difficulty distribution
            difficulties = [cat.difficulty for cat in puzzle.solution if cat.difficulty]
            if len(difficulties) == 4:
                difficulty_values = [d.value for d in difficulties]
                expected = {"YELLOW", "GREEN", "BLUE", "PURPLE"}
                actual = set(difficulty_values)
                
                if actual != expected:
                    validation_results["warnings"].append(f"Difficulty distribution: {actual}, expected: {expected}")
            else:
                validation_results["warnings"].append("Not all categories have assigned difficulties")
            
            # Check processing flags
            for category in puzzle.solution:
                if not category.processed_by_linguist:
                    validation_results["warnings"].append(f"Category '{category.category_name}' not processed by Linguist")
                if not category.processed_by_fact_checker:
                    validation_results["warnings"].append(f"Category '{category.category_name}' not processed by Fact-Checker")
                if not category.processed_by_trickster:
                    validation_results["warnings"].append(f"Category '{category.category_name}' not processed by Trickster")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating puzzle: {e}")
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
            return validation_results
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get detailed pipeline performance metrics."""
        return {
            "generation_metrics": self.stats,
            "component_health": {
                "knowledge_graph": self.knowledge_graph.health_check(),
                "cache": self.cache_manager.health_check(),
                "vector_store": len(self.vector_store.entities) > 0
            },
            "cache_stats": self.cache_manager.get_cache_stats(),
            "knowledge_graph_stats": self.knowledge_graph.get_statistics(),
            "vector_store_stats": self.vector_store.get_statistics()
        }