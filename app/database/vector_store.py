"""Vector store implementation for semantic similarity search."""

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

from ..models.entities import KnowledgeGraphEntity
from ..config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for semantic similarity search using sentence transformers."""
    
    def __init__(self, model_name: str = None):
        """Initialize the vector store with embedding model."""
        self.model_name = model_name or settings.default_embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.entities: Dict[str, KnowledgeGraphEntity] = {}
        
    def add_entity(self, entity: KnowledgeGraphEntity) -> bool:
        """Add an entity to the vector store with its embedding."""
        try:
            # Create text representation for embedding
            text_repr = self._entity_to_text(entity)
            
            # Generate embedding if not already present
            if entity.embedding is None:
                embedding = self.model.encode([text_repr])[0]
                entity.embedding = embedding.tolist()
            else:
                embedding = np.array(entity.embedding)
            
            # Store in memory
            self.embeddings[entity.entity_id] = embedding
            self.entities[entity.entity_id] = entity
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity {entity.entity_id} to vector store: {e}")
            return False
    
    def add_entities(self, entities: List[KnowledgeGraphEntity]) -> int:
        """Add multiple entities to the vector store."""
        success_count = 0
        
        # Prepare texts for batch embedding
        texts = []
        entity_ids = []
        entities_to_embed = []
        
        for entity in entities:
            if entity.embedding is None:
                text_repr = self._entity_to_text(entity)
                texts.append(text_repr)
                entity_ids.append(entity.entity_id)
                entities_to_embed.append(entity)
            else:
                # Entity already has embedding
                embedding = np.array(entity.embedding)
                self.embeddings[entity.entity_id] = embedding
                self.entities[entity.entity_id] = entity
                success_count += 1
        
        # Batch generate embeddings for entities without them
        if texts:
            try:
                embeddings = self.model.encode(texts)
                
                for i, entity in enumerate(entities_to_embed):
                    embedding = embeddings[i]
                    entity.embedding = embedding.tolist()
                    
                    self.embeddings[entity.entity_id] = embedding
                    self.entities[entity.entity_id] = entity
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
        
        return success_count
    
    def search_similar(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[KnowledgeGraphEntity, float]]:
        """Search for entities similar to the query text."""
        try:
            if not self.embeddings:
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for entity_id, entity_embedding in self.embeddings.items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    entity_embedding.reshape(1, -1)
                )[0, 0]
                
                if similarity >= threshold:
                    similarities.append((entity_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            results = []
            for entity_id, similarity in similarities[:top_k]:
                entity = self.entities[entity_id]
                results.append((entity, float(similarity)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar entities: {e}")
            return []
    
    def find_most_similar_group(self, entities: List[KnowledgeGraphEntity]) -> List[Tuple[KnowledgeGraphEntity, float]]:
        """Find the most semantically similar group within a list of entities."""
        try:
            if len(entities) < 2:
                return []
            
            # Get embeddings for all entities
            embeddings = []
            for entity in entities:
                if entity.entity_id in self.embeddings:
                    embeddings.append(self.embeddings[entity.entity_id])
                else:
                    # Generate embedding if not in store
                    text_repr = self._entity_to_text(entity)
                    embedding = self.model.encode([text_repr])[0]
                    embeddings.append(embedding)
            
            if len(embeddings) != len(entities):
                return []
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find the group of 4 entities with highest average pairwise similarity
            best_group = None
            best_avg_similarity = -1
            
            # Try all combinations of 4 entities
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    for k in range(j+1, len(entities)):
                        for l in range(k+1, len(entities)):
                            indices = [i, j, k, l]
                            
                            # Calculate average pairwise similarity for this group
                            similarities = []
                            for x in range(len(indices)):
                                for y in range(x+1, len(indices)):
                                    similarities.append(similarity_matrix[indices[x]][indices[y]])
                            
                            avg_similarity = np.mean(similarities)
                            
                            if avg_similarity > best_avg_similarity:
                                best_avg_similarity = avg_similarity
                                best_group = [entities[idx] for idx in indices]
            
            if best_group:
                return [(entity, best_avg_similarity) for entity in best_group]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error finding most similar group: {e}")
            return []
    
    def calculate_category_cohesion(self, words: List[str]) -> float:
        """Calculate semantic cohesion score for a category of words."""
        try:
            if len(words) < 2:
                return 0.0
            
            # Generate embeddings for words
            embeddings = self.model.encode(words)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    similarity = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0, 0]
                    similarities.append(similarity)
            
            # Return average similarity
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.error(f"Error calculating category cohesion: {e}")
            return 0.0
    
    def find_red_herring_candidates(self, category_words: List[str], candidate_words: List[str]) -> List[Tuple[str, float]]:
        """Find words that could serve as red herrings for a category."""
        try:
            # Get category centroid
            category_embeddings = self.model.encode(category_words)
            category_centroid = np.mean(category_embeddings, axis=0)
            
            # Calculate similarity of each candidate to the category
            candidate_embeddings = self.model.encode(candidate_words)
            
            results = []
            for i, word in enumerate(candidate_words):
                similarity = cosine_similarity(
                    category_centroid.reshape(1, -1),
                    candidate_embeddings[i].reshape(1, -1)
                )[0, 0]
                results.append((word, float(similarity)))
            
            # Sort by similarity (higher is better for red herrings)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding red herring candidates: {e}")
            return []
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeGraphEntity]:
        """Get an entity from the vector store."""
        return self.entities.get(entity_id)
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the vector store."""
        try:
            if entity_id in self.embeddings:
                del self.embeddings[entity_id]
            if entity_id in self.entities:
                del self.entities[entity_id]
            return True
        except Exception as e:
            logger.error(f"Error removing entity {entity_id}: {e}")
            return False
    
    def save_to_file(self, filepath: str) -> bool:
        """Save the vector store to a file."""
        try:
            data = {
                'model_name': self.model_name,
                'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
                'entities': {k: v.dict() for k, v in self.entities.items()}
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store to {filepath}: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load the vector store from a file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.model_name = data['model_name']
            if self.model_name != self.model.model_name:
                logger.warning(f"Loading embeddings from different model: {self.model_name}")
            
            self.embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
            
            # Reconstruct entities
            from ..models.entities import KnowledgeGraphEntity
            self.entities = {}
            for entity_id, entity_data in data['entities'].items():
                entity = KnowledgeGraphEntity(**entity_data)
                self.entities[entity_id] = entity
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store from {filepath}: {e}")
            return False
    
    def _entity_to_text(self, entity: KnowledgeGraphEntity) -> str:
        """Convert entity to text representation for embedding."""
        text_parts = [entity.entity_name]
        
        if entity.description:
            text_parts.append(entity.description)
        
        if entity.aliases:
            text_parts.extend(entity.aliases)
        
        if entity.tags:
            text_parts.extend(entity.tags)
        
        # Add some attributes as text
        for key, value in entity.attributes.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, list) and value and isinstance(value[0], str):
                text_parts.append(f"{key}: {', '.join(value)}")
        
        return " ".join(text_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_entities": len(self.entities),
            "total_embeddings": len(self.embeddings),
            "model_name": self.model_name,
            "embedding_dimension": len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        self.embeddings.clear()
        self.entities.clear()