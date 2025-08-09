"""Fact-Checker Agent for verifying factual accuracy using RAG."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import wikipedia
import requests
from datetime import datetime

from .base import BaseAgent
from ..models.puzzles import Puzzle, PuzzleCategory
from ..database.knowledge_graph import KnowledgeGraphDB
from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class FactCheckerAgent(BaseAgent):
    """Fact-Checker Agent that verifies factual accuracy using Retrieval-Augmented Generation."""
    
    def __init__(self, knowledge_graph: KnowledgeGraphDB, vector_store: VectorStore, **kwargs):
        """Initialize the Fact-Checker Agent."""
        super().__init__(**kwargs)
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify factual accuracy of puzzle categories."""
        self.validate_input(input_data, ["puzzle"])
        
        try:
            puzzle_data = input_data["puzzle"]
            if isinstance(puzzle_data, dict):
                puzzle = Puzzle(**puzzle_data)
            else:
                puzzle = puzzle_data
            
            # Verify each category
            verified_puzzle = self._verify_puzzle_facts(puzzle)
            
            return {
                "success": True,
                "puzzle": verified_puzzle.dict(),
                "fact_check_results": self._get_fact_check_summary(verified_puzzle),
                "metadata": self.get_agent_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in Fact-Checker Agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": self.get_agent_metadata()
            }
    
    def _verify_puzzle_facts(self, puzzle: Puzzle) -> Puzzle:
        """Verify factual accuracy for all categories in the puzzle."""
        
        verified_puzzle = puzzle.copy(deep=True)
        all_fact_check_results = []
        
        for category in verified_puzzle.solution:
            # Verify this category
            fact_check_result = self._verify_category_facts(category)
            category.fact_check_results = fact_check_result
            category.processed_by_fact_checker = True
            
            all_fact_check_results.append({
                "category": category.category_name,
                "result": fact_check_result
            })
        
        # Update metadata
        if not verified_puzzle.generation_metadata.fact_checker_metadata:
            verified_puzzle.generation_metadata.fact_checker_metadata = {}
        
        verified_puzzle.generation_metadata.fact_checker_metadata.update({
            "verification_timestamp": datetime.utcnow().isoformat(),
            "categories_verified": len(verified_puzzle.solution),
            "fact_check_results": all_fact_check_results
        })
        
        return verified_puzzle
    
    def _verify_category_facts(self, category: PuzzleCategory) -> Dict[str, Any]:
        """Verify factual accuracy of a single category."""
        
        logger.info(f"Fact-checking category: {category.category_name}")
        
        # Generate verification queries for this category
        verification_queries = self._generate_verification_queries(category)
        
        # Gather evidence for each query
        evidence_results = []
        for query in verification_queries:
            evidence = self._gather_evidence(query)
            evidence_results.append({
                "query": query,
                "evidence": evidence
            })
        
        # Use LLM to analyze evidence and make verification decision
        verification_result = self._analyze_evidence_with_llm(category, evidence_results)
        
        return {
            "category_name": category.category_name,
            "verification_queries": verification_queries,
            "evidence_gathered": evidence_results,
            "verification_result": verification_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_verification_queries(self, category: PuzzleCategory) -> List[str]:
        """Generate specific verification queries for a category."""
        
        words_list = self.format_word_list(category.words)
        
        prompt = self.create_system_prompt(
            "a fact-checking expert specializing in UK cultural knowledge",
            [
                "Generate specific, verifiable queries to fact-check puzzle categories",
                "Focus on connections that can be verified against reliable sources",
                "Ensure queries are specific enough to get definitive answers"
            ]
        )
        
        prompt += f"""
Generate 3-5 specific fact-checking queries for this puzzle category:

Category: "{category.category_name}"
Words: {words_list}
Category Type: {category.category_type.value}

Create queries that verify:
1. Whether each word truly belongs to the claimed category
2. Whether the category name accurately describes the connection
3. Whether there are any factual errors or misconceptions

Make queries specific and searchable. Examples:
- "Is [specific character] actually from [specific TV show]?"
- "Is [specific location] actually in [claimed region]?"
- "Was [specific person] actually a [claimed profession/role]?"

Respond with JSON:
{{
    "queries": [
        "Specific verification query 1",
        "Specific verification query 2",
        "Specific verification query 3"
    ],
    "focus_areas": ["areas of concern to verify"]
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.3)
            parsed_response = self.parse_json_response(response)
            
            queries = parsed_response.get("queries", [])
            return queries[:5]  # Limit to 5 queries max
            
        except Exception as e:
            logger.error(f"Error generating verification queries: {e}")
            # Fallback to generic queries
            return [f"Verify that '{word}' belongs to category '{category.category_name}'" for word in category.words[:3]]
    
    def _gather_evidence(self, query: str) -> Dict[str, Any]:
        """Gather evidence for a verification query from multiple sources."""
        
        evidence = {
            "query": query,
            "sources": [],
            "reliability_score": 0.0
        }
        
        # Source 1: Wikipedia search
        wikipedia_evidence = self._search_wikipedia(query)
        if wikipedia_evidence:
            evidence["sources"].append(wikipedia_evidence)
        
        # Source 2: Knowledge graph lookup
        kg_evidence = self._search_knowledge_graph(query)
        if kg_evidence:
            evidence["sources"].append(kg_evidence)
        
        # Source 3: Vector similarity search
        vector_evidence = self._search_vector_store(query)
        if vector_evidence:
            evidence["sources"].append(vector_evidence)
        
        # Calculate reliability score based on source agreement
        evidence["reliability_score"] = self._calculate_reliability_score(evidence["sources"])
        
        return evidence
    
    def _search_wikipedia(self, query: str) -> Optional[Dict[str, Any]]:
        """Search Wikipedia for evidence."""
        
        try:
            # Search for relevant pages
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return None
            
            # Get summary from the most relevant page
            try:
                page = wikipedia.page(search_results[0])
                summary = page.summary[:500]  # First 500 characters
                
                return {
                    "source": "wikipedia",
                    "title": page.title,
                    "url": page.url,
                    "content": summary,
                    "reliability": 0.8  # Wikipedia is generally reliable
                }
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation by picking the first option
                if e.options:
                    page = wikipedia.page(e.options[0])
                    summary = page.summary[:500]
                    
                    return {
                        "source": "wikipedia",
                        "title": page.title,
                        "url": page.url,
                        "content": summary,
                        "reliability": 0.7  # Slightly lower due to disambiguation
                    }
            
        except Exception as e:
            logger.warning(f"Wikipedia search failed for query '{query}': {e}")
        
        return None
    
    def _search_knowledge_graph(self, query: str) -> Optional[Dict[str, Any]]:
        """Search the knowledge graph for evidence."""
        
        try:
            # Extract key terms from query
            key_terms = self._extract_key_terms_from_query(query)
            
            if not key_terms:
                return None
            
            # Search for entities matching key terms
            relevant_entities = []
            for term in key_terms:
                entities = self.knowledge_graph.query_entities(
                    type("EntityQuery", (), {"search_text": term, "limit": 5})()
                )
                relevant_entities.extend(entities)
            
            if not relevant_entities:
                return None
            
            # Format entity information
            entity_info = []
            for entity in relevant_entities[:3]:  # Limit to top 3
                info = {
                    "name": entity.entity_name,
                    "type": entity.entity_type.value,
                    "description": entity.description,
                    "cultural_tier": entity.cultural_specificity_tier.value
                }
                entity_info.append(info)
            
            return {
                "source": "knowledge_graph",
                "entities_found": len(entity_info),
                "entities": entity_info,
                "reliability": 0.9  # Our knowledge graph should be highly reliable
            }
            
        except Exception as e:
            logger.warning(f"Knowledge graph search failed for query '{query}': {e}")
        
        return None
    
    def _search_vector_store(self, query: str) -> Optional[Dict[str, Any]]:
        """Search the vector store for semantically similar entities."""
        
        try:
            similar_entities = self.vector_store.search_similar(query, top_k=3)
            
            if not similar_entities:
                return None
            
            entity_matches = []
            for entity, similarity in similar_entities:
                entity_matches.append({
                    "name": entity.entity_name,
                    "similarity_score": similarity,
                    "type": entity.entity_type.value
                })
            
            return {
                "source": "vector_similarity",
                "matches_found": len(entity_matches),
                "matches": entity_matches,
                "reliability": 0.6  # Vector similarity is less definitive for fact-checking
            }
            
        except Exception as e:
            logger.warning(f"Vector store search failed for query '{query}': {e}")
        
        return None
    
    def _extract_key_terms_from_query(self, query: str) -> List[str]:
        """Extract key terms from a verification query."""
        
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {"is", "are", "was", "were", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "that", "this", "these", "those", "actually", "really", "truly"}
        
        # Extract words, keeping those in quotes and proper nouns
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]+\b', query)
        
        # Filter out stop words and short words
        key_terms = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _calculate_reliability_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate overall reliability score based on source agreement."""
        
        if not sources:
            return 0.0
        
        # Weight by source reliability and number of sources
        total_reliability = sum(source.get("reliability", 0.5) for source in sources)
        avg_reliability = total_reliability / len(sources)
        
        # Bonus for multiple sources
        source_bonus = min(0.2, len(sources) * 0.05)
        
        return min(1.0, avg_reliability + source_bonus)
    
    def _analyze_evidence_with_llm(self, category: PuzzleCategory, evidence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to analyze gathered evidence and make verification decision."""
        
        words_list = self.format_word_list(category.words)
        
        # Format evidence for the prompt
        evidence_summary = []
        for result in evidence_results:
            query = result["query"]
            evidence = result["evidence"]
            
            sources_info = []
            for source in evidence.get("sources", []):
                source_type = source.get("source", "unknown")
                reliability = source.get("reliability", 0.0)
                content = source.get("content", source.get("entities", source.get("matches", "")))
                sources_info.append(f"- {source_type} (reliability: {reliability:.1f}): {str(content)[:200]}...")
            
            evidence_summary.append(f"Query: {query}\nEvidence:\n" + "\n".join(sources_info))
        
        evidence_text = "\n\n".join(evidence_summary)
        
        prompt = self.create_system_prompt(
            "an expert fact-checker specializing in UK cultural knowledge",
            [
                "Analyze evidence to verify factual accuracy of puzzle categories",
                "Be conservative - flag potential issues even if evidence is mixed",
                "Consider cultural context and UK-specific knowledge",
                "Identify any factual errors or weak connections"
            ]
        )
        
        prompt += f"""
Analyze the evidence to verify this puzzle category:

Category: "{category.category_name}"
Words: {words_list}
Category Type: {category.category_type.value}

Evidence gathered:
{evidence_text}

Based on the evidence, determine:
1. Are all words correctly categorized?
2. Is the category name accurate?
3. Are there any factual errors?
4. What is the confidence level in this categorization?

Respond with JSON:
{{
    "verification_status": "verified" | "flagged" | "uncertain",
    "confidence_score": 0.85,
    "issues_found": ["list of specific issues, if any"],
    "incorrect_words": ["words that don't belong, if any"],
    "suggested_corrections": ["suggested fixes, if any"],
    "reasoning": "Detailed explanation of the verification decision"
}}
"""
        
        try:
            response = self.call_llm_with_cache(prompt, temperature=0.2)
            parsed_response = self.parse_json_response(response)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error analyzing evidence with LLM: {e}")
            return {
                "verification_status": "uncertain",
                "confidence_score": 0.0,
                "issues_found": [f"LLM analysis failed: {str(e)}"],
                "reasoning": "Could not complete fact-checking analysis due to technical error"
            }
    
    def _get_fact_check_summary(self, puzzle: Puzzle) -> Dict[str, Any]:
        """Generate a summary of fact-checking results."""
        
        total_categories = len(puzzle.solution)
        verified_count = 0
        flagged_count = 0
        uncertain_count = 0
        
        category_summaries = []
        
        for category in puzzle.solution:
            if category.fact_check_results:
                verification_result = category.fact_check_results.get("verification_result", {})
                status = verification_result.get("verification_status", "uncertain")
                confidence = verification_result.get("confidence_score", 0.0)
                
                if status == "verified":
                    verified_count += 1
                elif status == "flagged":
                    flagged_count += 1
                else:
                    uncertain_count += 1
                
                category_summaries.append({
                    "category": category.category_name,
                    "status": status,
                    "confidence": confidence,
                    "issues": verification_result.get("issues_found", [])
                })
        
        return {
            "total_categories": total_categories,
            "verified": verified_count,
            "flagged": flagged_count,
            "uncertain": uncertain_count,
            "overall_reliability": verified_count / total_categories if total_categories > 0 else 0.0,
            "category_details": category_summaries
        }