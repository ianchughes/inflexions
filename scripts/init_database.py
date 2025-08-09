#!/usr/bin/env python3
"""Database initialization script for the AI Editorial Engine."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.config import settings
from app.database import KnowledgeGraphDB, VectorStore, CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_knowledge_graph():
    """Initialize the knowledge graph database."""
    logger.info("Initializing Knowledge Graph database...")
    
    try:
        kg = KnowledgeGraphDB()
        
        # Test connection
        if kg.health_check():
            logger.info("Knowledge Graph database connection successful")
            stats = kg.get_statistics()
            logger.info(f"Current stats: {stats}")
        else:
            logger.error("Knowledge Graph database connection failed")
            return False
        
        kg.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Knowledge Graph: {e}")
        return False


def init_cache():
    """Initialize the cache."""
    logger.info("Initializing Cache...")
    
    try:
        cache = CacheManager()
        
        # Test connection
        if cache.health_check():
            logger.info("Cache connection successful")
            stats = cache.get_cache_stats()
            logger.info(f"Cache stats: {stats}")
        else:
            logger.error("Cache connection failed")
            return False
        
        cache.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Cache: {e}")
        return False


def init_vector_store():
    """Initialize the vector store."""
    logger.info("Initializing Vector Store...")
    
    try:
        vs = VectorStore()
        stats = vs.get_statistics()
        logger.info(f"Vector Store initialized. Stats: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Vector Store: {e}")
        return False


def main():
    """Main initialization function."""
    logger.info("Starting database initialization...")
    
    # Initialize components
    components = [
        ("Knowledge Graph", init_knowledge_graph),
        ("Cache", init_cache), 
        ("Vector Store", init_vector_store)
    ]
    
    success_count = 0
    for name, init_func in components:
        if init_func():
            success_count += 1
            logger.info(f"✓ {name} initialized successfully")
        else:
            logger.error(f"✗ {name} initialization failed")
    
    if success_count == len(components):
        logger.info("🎉 All components initialized successfully!")
        return 0
    else:
        logger.error(f"⚠️  {len(components) - success_count} components failed to initialize")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)