"""Cache manager implementation using Redis."""

import logging
import json
import pickle
from typing import Any, Optional, Dict, List
import redis
from datetime import timedelta

from ..config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager for the AI Editorial Engine."""
    
    def __init__(self, redis_url: str = None):
        """Initialize Redis connection."""
        self.redis_url = redis_url or settings.redis_url
        self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
        self.default_ttl = settings.cache_ttl_seconds
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache with optional TTL."""
        try:
            ttl = ttl or self.default_ttl
            
            # Serialize the value
            serialized_value = pickle.dumps(value)
            
            # Set in Redis with TTL
            result = self.redis_client.setex(key, ttl, serialized_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        try:
            serialized_value = self.redis_client.get(key)
            
            if serialized_value is None:
                return None
            
            # Deserialize the value
            value = pickle.loads(serialized_value)
            return value
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        try:
            result = self.redis_client.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking existence of cache key {key}: {e}")
            return False
    
    def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set a JSON-serializable value in the cache."""
        try:
            ttl = ttl or self.default_ttl
            json_value = json.dumps(value)
            result = self.redis_client.setex(key, ttl, json_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting JSON cache key {key}: {e}")
            return False
    
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a JSON value from the cache."""
        try:
            json_value = self.redis_client.get(key)
            
            if json_value is None:
                return None
            
            # Decode if bytes
            if isinstance(json_value, bytes):
                json_value = json_value.decode('utf-8')
            
            return json.loads(json_value)
            
        except Exception as e:
            logger.error(f"Error getting JSON cache key {key}: {e}")
            return None
    
    def set_list(self, key: str, values: List[Any], ttl: Optional[int] = None) -> bool:
        """Set a list of values in the cache using Redis list operations."""
        try:
            ttl = ttl or self.default_ttl
            
            # Delete existing list
            self.redis_client.delete(key)
            
            # Add values to list
            if values:
                serialized_values = [pickle.dumps(value) for value in values]
                self.redis_client.lpush(key, *serialized_values)
            
            # Set TTL
            self.redis_client.expire(key, ttl)
            return True
            
        except Exception as e:
            logger.error(f"Error setting list cache key {key}: {e}")
            return False
    
    def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get a list of values from the cache."""
        try:
            serialized_values = self.redis_client.lrange(key, start, end)
            
            values = []
            for serialized_value in serialized_values:
                try:
                    value = pickle.loads(serialized_value)
                    values.append(value)
                except Exception as e:
                    logger.warning(f"Error deserializing list item: {e}")
                    continue
            
            return values
            
        except Exception as e:
            logger.error(f"Error getting list cache key {key}: {e}")
            return []
    
    def cache_puzzle(self, puzzle_id: str, puzzle_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache puzzle data."""
        cache_key = f"puzzle:{puzzle_id}"
        return self.set_json(cache_key, puzzle_data, ttl)
    
    def get_cached_puzzle(self, puzzle_id: str) -> Optional[Dict[str, Any]]:
        """Get cached puzzle data."""
        cache_key = f"puzzle:{puzzle_id}"
        return self.get_json(cache_key)
    
    def cache_entity_query(self, query_hash: str, entities: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """Cache entity query results."""
        cache_key = f"entity_query:{query_hash}"
        return self.set_json(cache_key, {"entities": entities}, ttl)
    
    def get_cached_entity_query(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached entity query results."""
        cache_key = f"entity_query:{query_hash}"
        result = self.get_json(cache_key)
        return result.get("entities") if result else None
    
    def cache_embeddings(self, entity_id: str, embeddings: List[float], ttl: Optional[int] = None) -> bool:
        """Cache entity embeddings."""
        cache_key = f"embeddings:{entity_id}"
        return self.set_json(cache_key, {"embeddings": embeddings}, ttl)
    
    def get_cached_embeddings(self, entity_id: str) -> Optional[List[float]]:
        """Get cached entity embeddings."""
        cache_key = f"embeddings:{entity_id}"
        result = self.get_json(cache_key)
        return result.get("embeddings") if result else None
    
    def cache_difficulty_metrics(self, category_id: str, metrics: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache difficulty assessment metrics."""
        cache_key = f"difficulty_metrics:{category_id}"
        return self.set_json(cache_key, metrics, ttl)
    
    def get_cached_difficulty_metrics(self, category_id: str) -> Optional[Dict[str, Any]]:
        """Get cached difficulty assessment metrics."""
        cache_key = f"difficulty_metrics:{category_id}"
        return self.get_json(cache_key)
    
    def cache_agent_result(self, agent_name: str, input_hash: str, result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache agent processing results."""
        cache_key = f"agent_result:{agent_name}:{input_hash}"
        return self.set_json(cache_key, result, ttl)
    
    def get_cached_agent_result(self, agent_name: str, input_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached agent processing results."""
        cache_key = f"agent_result:{agent_name}:{input_hash}"
        return self.get_json(cache_key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0
    
    def get_ttl(self, key: str) -> int:
        """Get the time-to-live for a key."""
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -1
    
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend the TTL of a key."""
        try:
            current_ttl = self.get_ttl(key)
            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                return bool(self.redis_client.expire(key, new_ttl))
            return False
            
        except Exception as e:
            logger.error(f"Error extending TTL for key {key}: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            
            stats = {
                "total_keys": self.redis_client.dbsize(),
                "memory_used": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
            
            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total = hits + misses
            if total > 0:
                stats["hit_rate"] = hits / total
            else:
                stats["hit_rate"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_all(self) -> bool:
        """Clear all data from the cache."""
        try:
            self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()