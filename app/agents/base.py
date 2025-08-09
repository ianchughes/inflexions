"""Base agent class with common LLM functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import openai
import anthropic
from datetime import datetime
import hashlib

from ..config import settings
from ..database.cache import CacheManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all AI agents in the editorial pipeline."""
    
    def __init__(self, model_name: str = None, cache_manager: CacheManager = None):
        """Initialize the base agent."""
        self.model_name = model_name or settings.default_generation_model
        self.cache_manager = cache_manager or CacheManager()
        
        # Initialize LLM clients
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        
        if settings.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            self.anthropic_client = None
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results."""
        pass
    
    def call_llm(self, prompt: str, model: str = None, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Call the appropriate LLM based on model name."""
        model = model or self.model_name
        
        try:
            if model.startswith('gpt-'):
                return self._call_openai(prompt, model, max_tokens, temperature)
            elif model.startswith('claude-'):
                return self._call_anthropic(prompt, model, max_tokens, temperature)
            else:
                # Default to OpenAI
                return self._call_openai(prompt, model, max_tokens, temperature)
                
        except Exception as e:
            logger.error(f"Error calling LLM {model}: {e}")
            raise
    
    def _call_openai(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Call OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Call Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def call_llm_with_cache(self, prompt: str, cache_key: str = None, **kwargs) -> str:
        """Call LLM with caching support."""
        # Generate cache key if not provided
        if not cache_key:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            cache_key = f"llm_response:{self.model_name}:{prompt_hash}"
        
        # Check cache first
        cached_response = self.cache_manager.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached_response
        
        # Call LLM and cache response
        response = self.call_llm(prompt, **kwargs)
        self.cache_manager.set(cache_key, response)
        
        return response
    
    def get_agent_metadata(self) -> Dict[str, Any]:
        """Get metadata about this agent."""
        return {
            "agent_name": self.__class__.__name__,
            "model_name": self.model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that input data contains required keys."""
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        return True
    
    def create_system_prompt(self, role_description: str, guidelines: List[str] = None) -> str:
        """Create a system prompt for the agent."""
        prompt_parts = [
            f"You are {role_description}.",
            "",
            "Context: You are part of an AI Editorial Engine that creates 'UK Connections' puzzles.",
            "These are word puzzles where players must find groups of 4 related words from a 4x4 grid.",
            "The puzzles focus on UK culture, including TV shows, history, food, slang, and more.",
            ""
        ]
        
        if guidelines:
            prompt_parts.append("Guidelines:")
            for guideline in guidelines:
                prompt_parts.append(f"- {guideline}")
            prompt_parts.append("")
        
        return "\n".join(prompt_parts)
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling common formatting issues."""
        import json
        import re
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            # Find the first and last ``` markers
            start_idx = next(i for i, line in enumerate(lines) if line.startswith('```'))
            end_idx = next(i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith('```'))
            response = '\n'.join(lines[start_idx + 1:end_idx])
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response: {e}")
    
    def format_word_list(self, words: List[str]) -> str:
        """Format a list of words for display in prompts."""
        return ", ".join(f'"{word}"' for word in words)
    
    def calculate_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Calculate a hash of input data for caching."""
        import json
        # Convert to JSON string and hash
        json_str = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()