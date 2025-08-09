"""Configuration management for the AI Editorial Engine."""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Database Configuration
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    
    # Vector Database
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    weaviate_url: str = Field("http://localhost:8080", env="WEAVIATE_URL")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Application Settings
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Model Configuration
    default_generation_model: str = Field("gpt-4-turbo", env="DEFAULT_GENERATION_MODEL")
    default_embedding_model: str = Field("all-mpnet-base-v2", env="DEFAULT_EMBEDDING_MODEL")
    difficulty_assessment_model: str = Field("gpt-4o", env="DIFFICULTY_ASSESSMENT_MODEL")
    
    # Pipeline Settings
    max_generation_attempts: int = Field(3, env="MAX_GENERATION_ATTEMPTS")
    enable_human_review: bool = Field(True, env="ENABLE_HUMAN_REVIEW")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()