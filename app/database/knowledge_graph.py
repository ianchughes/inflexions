"""Knowledge Graph database implementation using Neo4j."""

import logging
from typing import List, Optional, Dict, Any, Set
from neo4j import GraphDatabase, basic_auth
import json

from ..models.entities import (
    KnowledgeGraphEntity, 
    EntityRelationship, 
    EntityQuery,
    EntityType,
    CulturalSpecificityTier
)
from ..config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphDB:
    """Neo4j-based implementation of the UK Cultural Knowledge Graph."""
    
    def __init__(self):
        """Initialize connection to Neo4j database."""
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=basic_auth(settings.neo4j_user, settings.neo4j_password)
        )
        self._create_constraints()
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
    
    def _create_constraints(self):
        """Create necessary constraints and indexes."""
        with self.driver.session() as session:
            # Create unique constraint on entity_id
            session.run("""
                CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
            """)
            
            # Create indexes for common queries
            session.run("""
                CREATE INDEX entity_name_index IF NOT EXISTS
                FOR (e:Entity) ON (e.entity_name)
            """)
            
            session.run("""
                CREATE INDEX entity_type_index IF NOT EXISTS
                FOR (e:Entity) ON (e.entity_type)
            """)
            
            session.run("""
                CREATE INDEX cultural_tier_index IF NOT EXISTS
                FOR (e:Entity) ON (e.cultural_specificity_tier)
            """)
    
    def create_entity(self, entity: KnowledgeGraphEntity) -> bool:
        """Create a new entity in the knowledge graph."""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.entity_name = $entity_name,
                    e.entity_type = $entity_type,
                    e.attributes = $attributes,
                    e.cultural_specificity_tier = $cultural_specificity_tier,
                    e.related_entities = $related_entities,
                    e.source = $source,
                    e.aliases = $aliases,
                    e.description = $description,
                    e.tags = $tags,
                    e.embedding = $embedding,
                    e.created_at = $created_at,
                    e.updated_at = $updated_at,
                    e.validation_status = $validation_status
                RETURN e
                """
                
                result = session.run(query, {
                    "entity_id": entity.entity_id,
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type.value,
                    "attributes": json.dumps(entity.attributes),
                    "cultural_specificity_tier": entity.cultural_specificity_tier.value,
                    "related_entities": entity.related_entities,
                    "source": entity.source,
                    "aliases": entity.aliases,
                    "description": entity.description,
                    "tags": entity.tags,
                    "embedding": entity.embedding,
                    "created_at": entity.created_at.isoformat(),
                    "updated_at": entity.updated_at.isoformat(),
                    "validation_status": entity.validation_status
                })
                
                return bool(result.single())
                
        except Exception as e:
            logger.error(f"Error creating entity {entity.entity_id}: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeGraphEntity]:
        """Retrieve an entity by ID."""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Entity {entity_id: $entity_id})
                RETURN e
                """
                
                result = session.run(query, {"entity_id": entity_id})
                record = result.single()
                
                if record:
                    entity_data = dict(record["e"])
                    return self._record_to_entity(entity_data)
                    
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving entity {entity_id}: {e}")
            return None
    
    def query_entities(self, query: EntityQuery) -> List[KnowledgeGraphEntity]:
        """Query entities based on search criteria."""
        try:
            with self.driver.session() as session:
                cypher_query = "MATCH (e:Entity) WHERE 1=1"
                parameters = {}
                
                # Add filters based on query parameters
                if query.entity_types:
                    cypher_query += " AND e.entity_type IN $entity_types"
                    parameters["entity_types"] = [et.value for et in query.entity_types]
                
                if query.cultural_specificity_tiers:
                    cypher_query += " AND e.cultural_specificity_tier IN $cultural_tiers"
                    parameters["cultural_tiers"] = [tier.value for tier in query.cultural_specificity_tiers]
                
                if query.search_text:
                    cypher_query += " AND (toLower(e.entity_name) CONTAINS toLower($search_text) OR toLower(e.description) CONTAINS toLower($search_text))"
                    parameters["search_text"] = query.search_text
                
                if query.tags:
                    cypher_query += " AND any(tag IN $tags WHERE tag IN e.tags)"
                    parameters["tags"] = query.tags
                
                cypher_query += " RETURN e ORDER BY e.entity_name"
                cypher_query += f" SKIP {query.offset} LIMIT {query.limit}"
                
                result = session.run(cypher_query, parameters)
                
                entities = []
                for record in result:
                    entity_data = dict(record["e"])
                    entity = self._record_to_entity(entity_data)
                    if entity:
                        entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Error querying entities: {e}")
            return []
    
    def get_entities_by_type(self, entity_type: EntityType, limit: int = 50) -> List[KnowledgeGraphEntity]:
        """Get entities of a specific type."""
        query = EntityQuery(entity_types=[entity_type], limit=limit)
        return self.query_entities(query)
    
    def get_entities_by_tier(self, tier: CulturalSpecificityTier, limit: int = 50) -> List[KnowledgeGraphEntity]:
        """Get entities of a specific cultural specificity tier."""
        query = EntityQuery(cultural_specificity_tiers=[tier], limit=limit)
        return self.query_entities(query)
    
    def create_relationship(self, relationship: EntityRelationship) -> bool:
        """Create a relationship between two entities."""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source:Entity {entity_id: $source_id})
                MATCH (target:Entity {entity_id: $target_id})
                MERGE (source)-[r:RELATED {type: $rel_type}]->(target)
                SET r.strength = $strength,
                    r.metadata = $metadata
                RETURN r
                """
                
                result = session.run(query, {
                    "source_id": relationship.source_entity_id,
                    "target_id": relationship.target_entity_id,
                    "rel_type": relationship.relationship_type,
                    "strength": relationship.strength,
                    "metadata": json.dumps(relationship.metadata)
                })
                
                return bool(result.single())
                
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False
    
    def get_related_entities(self, entity_id: str, relationship_types: Optional[List[str]] = None) -> List[KnowledgeGraphEntity]:
        """Get entities related to a given entity."""
        try:
            with self.driver.session() as session:
                if relationship_types:
                    query = """
                    MATCH (source:Entity {entity_id: $entity_id})-[r:RELATED]->(target:Entity)
                    WHERE r.type IN $rel_types
                    RETURN target
                    ORDER BY r.strength DESC
                    """
                    parameters = {"entity_id": entity_id, "rel_types": relationship_types}
                else:
                    query = """
                    MATCH (source:Entity {entity_id: $entity_id})-[r:RELATED]->(target:Entity)
                    RETURN target
                    ORDER BY r.strength DESC
                    """
                    parameters = {"entity_id": entity_id}
                
                result = session.run(query, parameters)
                
                entities = []
                for record in result:
                    entity_data = dict(record["target"])
                    entity = self._record_to_entity(entity_data)
                    if entity:
                        entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Error getting related entities for {entity_id}: {e}")
            return []
    
    def get_entities_for_category(self, category_type: str, count: int = 10) -> List[KnowledgeGraphEntity]:
        """Get entities suitable for a specific puzzle category type."""
        try:
            # Map category types to entity queries
            entity_type_mapping = {
                "CULTURAL_TV": [EntityType.TV_SHOW, EntityType.CHARACTER],
                "CULTURAL_FILM": [EntityType.MOVIE, EntityType.ACTOR],
                "CULTURAL_MUSIC": [EntityType.MUSIC_ARTIST, EntityType.SONG, EntityType.ALBUM],
                "CULTURAL_LITERATURE": [EntityType.BOOK, EntityType.AUTHOR],
                "CULTURAL_HISTORY": [EntityType.HISTORICAL_FIGURE, EntityType.HISTORICAL_EVENT],
                "GEOGRAPHICAL": [EntityType.LOCATION, EntityType.LANDMARK],
                "CULINARY": [EntityType.FOOD, EntityType.DRINK]
            }
            
            entity_types = entity_type_mapping.get(category_type, [])
            if not entity_types:
                # Default to all types if category not specifically mapped
                entity_types = list(EntityType)
            
            query = EntityQuery(entity_types=entity_types, limit=count)
            return self.query_entities(query)
            
        except Exception as e:
            logger.error(f"Error getting entities for category {category_type}: {e}")
            return []
    
    def search_by_semantic_similarity(self, query_text: str, limit: int = 10) -> List[KnowledgeGraphEntity]:
        """Search entities by semantic similarity (placeholder for vector search integration)."""
        # This would integrate with the vector store for semantic search
        # For now, fall back to text search
        query = EntityQuery(search_text=query_text, limit=limit)
        return self.query_entities(query)
    
    def get_random_entities(self, count: int = 16, tier_weights: Optional[Dict[CulturalSpecificityTier, float]] = None) -> List[KnowledgeGraphEntity]:
        """Get random entities weighted by cultural specificity tier."""
        try:
            with self.driver.session() as session:
                if tier_weights:
                    # Weighted random selection based on tiers
                    entities = []
                    for tier, weight in tier_weights.items():
                        tier_count = max(1, int(count * weight))
                        query = """
                        MATCH (e:Entity {cultural_specificity_tier: $tier})
                        RETURN e
                        ORDER BY rand()
                        LIMIT $limit
                        """
                        result = session.run(query, {"tier": tier.value, "limit": tier_count})
                        
                        for record in result:
                            entity_data = dict(record["e"])
                            entity = self._record_to_entity(entity_data)
                            if entity:
                                entities.append(entity)
                    
                    return entities[:count]
                else:
                    # Simple random selection
                    query = """
                    MATCH (e:Entity)
                    RETURN e
                    ORDER BY rand()
                    LIMIT $limit
                    """
                    result = session.run(query, {"limit": count})
                    
                    entities = []
                    for record in result:
                        entity_data = dict(record["e"])
                        entity = self._record_to_entity(entity_data)
                        if entity:
                            entities.append(entity)
                    
                    return entities
                    
        except Exception as e:
            logger.error(f"Error getting random entities: {e}")
            return []
    
    def _record_to_entity(self, record: Dict[str, Any]) -> Optional[KnowledgeGraphEntity]:
        """Convert Neo4j record to KnowledgeGraphEntity."""
        try:
            # Parse JSON fields
            attributes = json.loads(record.get("attributes", "{}"))
            
            entity = KnowledgeGraphEntity(
                entity_id=record["entity_id"],
                entity_name=record["entity_name"],
                entity_type=EntityType(record["entity_type"]),
                attributes=attributes,
                cultural_specificity_tier=CulturalSpecificityTier(record["cultural_specificity_tier"]),
                related_entities=record.get("related_entities", []),
                source=record.get("source"),
                aliases=record.get("aliases", []),
                description=record.get("description"),
                tags=record.get("tags", []),
                embedding=record.get("embedding"),
                validation_status=record.get("validation_status", "pending")
            )
            
            return entity
            
        except Exception as e:
            logger.error(f"Error converting record to entity: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            with self.driver.session() as session:
                stats = {}
                
                # Total entities
                result = session.run("MATCH (e:Entity) RETURN count(e) as total")
                stats["total_entities"] = result.single()["total"]
                
                # Entities by type
                result = session.run("""
                    MATCH (e:Entity) 
                    RETURN e.entity_type as type, count(e) as count
                    ORDER BY count DESC
                """)
                stats["entities_by_type"] = {record["type"]: record["count"] for record in result}
                
                # Entities by cultural tier
                result = session.run("""
                    MATCH (e:Entity) 
                    RETURN e.cultural_specificity_tier as tier, count(e) as count
                    ORDER BY tier
                """)
                stats["entities_by_tier"] = {record["tier"]: record["count"] for record in result}
                
                # Total relationships
                result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as total")
                stats["total_relationships"] = result.single()["total"]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                return bool(result.single())
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False