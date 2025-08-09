# AI Editorial Engine for 'UK Connections'

A sophisticated procedural content generation (PCG) system designed to automate the creation, curation, and validation of 'UK Connections' puzzles. The system uses a multi-stage pipeline of specialized AI agents that interact with a central UK-centric knowledge base.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   Generation    │    │    Curation     │    │   Assessment    │
│    (UK-CKG)     │───▶│  Creator Agent  │───▶│ Editorial Agents│───▶│  Judge Agent    │
│                 │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Pipeline Stages

1. **Data Layer (UK-CKG)**: UK Cultural Knowledge Graph containing structured data on UK culture
2. **Generation Stage**: 'Creator' Agent generates raw candidate puzzles using Tree of Thoughts prompting
3. **Curation Stage**: Sequential refinement by specialized agents:
   - **Trickster Agent**: Introduces deliberate misdirection
   - **Linguist Agent**: Refines category names and ensures linguistic consistency
   - **Fact-Checker Agent**: Verifies factual accuracy using RAG
4. **Assessment Stage**: 'Judge' Agent assigns final difficulty ratings (Yellow, Green, Blue, Purple)

## Features

- **Multi-Agent Pipeline**: Specialized AI agents for different aspects of puzzle creation
- **UK Cultural Knowledge Graph**: Comprehensive database of UK-specific cultural references
- **Cultural Specificity Tiers**: 4-tier system for controlling puzzle accessibility
- **Hybrid Difficulty Assessment**: Combines quantitative metrics with qualitative LLM judgment
- **Tree of Thoughts Prompting**: Advanced reasoning for creative puzzle generation
- **Retrieval-Augmented Generation**: Fact-checking to prevent AI hallucinations

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Initialize Database**
   ```bash
   python scripts/init_database.py
   ```

4. **Populate Knowledge Graph**
   ```bash
   python scripts/populate_knowledge_graph.py
   ```

5. **Run the API Server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `POST /api/v1/puzzles/generate` - Generate a new puzzle
- `GET /api/v1/puzzles/{puzzle_id}` - Retrieve a specific puzzle
- `GET /api/v1/knowledge-graph/entities` - Query knowledge graph entities
- `POST /api/v1/knowledge-graph/entities` - Add new entities to knowledge graph

## Configuration

The system supports multiple database backends:
- **Neo4j** for graph database functionality
- **Pinecone/Weaviate** for vector search
- **Redis** for caching and session management

## Cultural Specificity Tiers

- **Tier 1 (Global/Pan-UK)**: Universally known (e.g., The Beatles, Shakespeare)
- **Tier 2 (Broadly British)**: Household names in the UK (e.g., Only Fools and Horses, Marmite)
- **Tier 3 (Culturally Attuned)**: Requires deeper cultural fluency (e.g., The Thick of It characters)
- **Tier 4 (Niche/Regional)**: Highly specific knowledge (e.g., Glaswegian slang, regional names)

## Development

```bash
# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy .
```

## License

MIT License - see LICENSE file for details.