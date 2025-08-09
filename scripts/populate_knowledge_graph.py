#!/usr/bin/env python3
"""Script to populate the knowledge graph with sample UK cultural data."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
from app.database import KnowledgeGraphDB, VectorStore
from app.models.entities import KnowledgeGraphEntity, EntityType, CulturalSpecificityTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_entities():
    """Create sample entities representing UK cultural knowledge."""
    
    entities = []
    
    # British TV Shows
    entities.extend([
        KnowledgeGraphEntity(
            entity_id="uk-tv-only-fools-and-horses",
            entity_name="Only Fools and Horses",
            entity_type=EntityType.TV_SHOW,
            attributes={
                "genre": "Sitcom",
                "channel": "BBC One",
                "creator": "John Sullivan",
                "years": "1981-2003",
                "main_characters": ["Del Boy", "Rodney", "Grandad", "Uncle Albert"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Classic British sitcom about South London market traders",
            tags=["comedy", "family", "working_class", "london"],
            source="https://en.wikipedia.org/wiki/Only_Fools_and_Horses"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-tv-fawlty-towers",
            entity_name="Fawlty Towers",
            entity_type=EntityType.TV_SHOW,
            attributes={
                "genre": "Sitcom",
                "channel": "BBC Two",
                "creator": "John Cleese",
                "years": "1975-1979",
                "main_characters": ["Basil Fawlty", "Sybil Fawlty", "Manuel", "Polly"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="British sitcom about a rude hotel owner",
            tags=["comedy", "hotel", "monty_python"],
            source="https://en.wikipedia.org/wiki/Fawlty_Towers"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-tv-doctor-who",
            entity_name="Doctor Who",
            entity_type=EntityType.TV_SHOW,
            attributes={
                "genre": "Science Fiction",
                "channel": "BBC One",
                "creator": "Sydney Newman",
                "years": "1963-present",
                "main_characters": ["The Doctor", "Companion"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            description="Long-running British science fiction television series",
            tags=["sci_fi", "time_travel", "bbc"],
            source="https://en.wikipedia.org/wiki/Doctor_Who"
        )
    ])
    
    # British Characters
    entities.extend([
        KnowledgeGraphEntity(
            entity_id="uk-char-del-boy",
            entity_name="Del Boy",
            entity_type=EntityType.CHARACTER,
            attributes={
                "full_name": "Derek Edward Trotter",
                "show": "Only Fools and Horses",
                "actor": "David Jason",
                "catchphrases": ["This time next year we'll be millionaires", "Lovely jubbly"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Main character from Only Fools and Horses",
            tags=["comedy", "entrepreneur", "cockney"],
            source="https://en.wikipedia.org/wiki/Del_Boy"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-char-basil-fawlty",
            entity_name="Basil Fawlty",
            entity_type=EntityType.CHARACTER,
            attributes={
                "show": "Fawlty Towers",
                "actor": "John Cleese",
                "occupation": "Hotel Owner",
                "personality": "Rude, sarcastic, incompetent"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Main character from Fawlty Towers",
            tags=["comedy", "hotel", "rude"],
            source="https://en.wikipedia.org/wiki/Basil_Fawlty"
        )
    ])
    
    # British Authors
    entities.extend([
        KnowledgeGraphEntity(
            entity_id="uk-author-shakespeare",
            entity_name="Shakespeare",
            entity_type=EntityType.AUTHOR,
            attributes={
                "full_name": "William Shakespeare",
                "period": "Elizabethan",
                "famous_works": ["Hamlet", "Romeo and Juliet", "Macbeth", "Othello"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            description="Greatest English playwright and poet",
            tags=["literature", "playwright", "elizabethan"],
            source="https://en.wikipedia.org/wiki/William_Shakespeare"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-author-dickens",
            entity_name="Dickens",
            entity_type=EntityType.AUTHOR,
            attributes={
                "full_name": "Charles Dickens",
                "period": "Victorian",
                "famous_works": ["Oliver Twist", "Great Expectations", "A Christmas Carol", "David Copperfield"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            description="Victorian English novelist",
            tags=["literature", "victorian", "social_reform"],
            source="https://en.wikipedia.org/wiki/Charles_Dickens"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-author-austen",
            entity_name="Austen",
            entity_type=EntityType.AUTHOR,
            attributes={
                "full_name": "Jane Austen",
                "period": "Regency",
                "famous_works": ["Pride and Prejudice", "Emma", "Sense and Sensibility", "Mansfield Park"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            description="English Regency novelist",
            tags=["literature", "romance", "regency"],
            source="https://en.wikipedia.org/wiki/Jane_Austen"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-author-christie",
            entity_name="Christie",
            entity_type=EntityType.AUTHOR,
            attributes={
                "full_name": "Agatha Christie",
                "genre": "Crime/Mystery",
                "famous_works": ["Murder on the Orient Express", "The Murder of Roger Ackroyd", "And Then There Were None"],
                "famous_characters": ["Hercule Poirot", "Miss Marple"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            description="British crime novelist",
            tags=["crime", "mystery", "detective"],
            source="https://en.wikipedia.org/wiki/Agatha_Christie"
        )
    ])
    
    # British Food
    entities.extend([
        KnowledgeGraphEntity(
            entity_id="uk-food-bacon",
            entity_name="Bacon",
            entity_type=EntityType.FOOD,
            attributes={
                "category": "Meat",
                "meal_context": "Full English Breakfast",
                "preparation": "Fried or grilled"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Cured pork, essential part of Full English Breakfast",
            tags=["breakfast", "meat", "pork"],
            source="https://en.wikipedia.org/wiki/Bacon"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-food-eggs",
            entity_name="Eggs",
            entity_type=EntityType.FOOD,
            attributes={
                "category": "Protein",
                "meal_context": "Full English Breakfast",
                "preparation": "Fried, scrambled, or poached"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Chicken eggs, breakfast staple",
            tags=["breakfast", "protein"],
            source="https://en.wikipedia.org/wiki/Egg_as_food"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-food-sausages",
            entity_name="Sausages",
            entity_type=EntityType.FOOD,
            attributes={
                "category": "Meat",
                "meal_context": "Full English Breakfast",
                "type": "Pork sausages"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Pork sausages, part of traditional breakfast",
            tags=["breakfast", "meat", "pork"],
            source="https://en.wikipedia.org/wiki/Sausage"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-food-beans",
            entity_name="Beans",
            entity_type=EntityType.FOOD,
            attributes={
                "category": "Vegetable",
                "meal_context": "Full English Breakfast",
                "type": "Baked beans in tomato sauce"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Baked beans in tomato sauce",
            tags=["breakfast", "vegetable", "beans"],
            source="https://en.wikipedia.org/wiki/Baked_beans"
        )
    ])
    
    # British Locations
    entities.extend([
        KnowledgeGraphEntity(
            entity_id="uk-location-london",
            entity_name="London",
            entity_type=EntityType.LOCATION,
            attributes={
                "type": "Capital City",
                "country": "England",
                "population": "9 million",
                "famous_landmarks": ["Big Ben", "Tower Bridge", "Buckingham Palace"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.GLOBAL_PAN_UK,
            description="Capital city of England and the UK",
            tags=["capital", "city", "england"],
            source="https://en.wikipedia.org/wiki/London"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-location-edinburgh",
            entity_name="Edinburgh",
            entity_type=EntityType.LOCATION,
            attributes={
                "type": "Capital City",
                "country": "Scotland", 
                "population": "500,000",
                "famous_landmarks": ["Edinburgh Castle", "Royal Mile", "Arthur's Seat"]
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Capital city of Scotland",
            tags=["capital", "city", "scotland"],
            source="https://en.wikipedia.org/wiki/Edinburgh"
        )
    ])
    
    # Words that can follow "BLACK"
    entities.extend([
        KnowledgeGraphEntity(
            entity_id="uk-word-cab",
            entity_name="Cab",
            entity_type=EntityType.OTHER,
            attributes={
                "context": "Black cab - London taxi",
                "full_phrase": "Black cab"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="London taxi (black cab)",
            tags=["transport", "london", "taxi"],
            source="https://en.wikipedia.org/wiki/Hackney_carriage"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-word-pudding",
            entity_name="Pudding",
            entity_type=EntityType.FOOD,
            attributes={
                "context": "Black pudding - blood sausage",
                "full_phrase": "Black pudding"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Blood sausage (black pudding)",
            tags=["food", "sausage", "breakfast"],
            source="https://en.wikipedia.org/wiki/Black_pudding"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-word-belt",
            entity_name="Belt",
            entity_type=EntityType.OTHER,
            attributes={
                "context": "Black belt - martial arts",
                "full_phrase": "Black belt"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Martial arts rank (black belt)",
            tags=["martial_arts", "achievement"],
            source="https://en.wikipedia.org/wiki/Black_belt_(martial_arts)"
        ),
        
        KnowledgeGraphEntity(
            entity_id="uk-word-sheep",
            entity_name="Sheep",
            entity_type=EntityType.OTHER,
            attributes={
                "context": "Black sheep - family outcast",
                "full_phrase": "Black sheep"
            },
            cultural_specificity_tier=CulturalSpecificityTier.BROADLY_BRITISH,
            description="Family outcast (black sheep)",
            tags=["idiom", "family"],
            source="https://en.wikipedia.org/wiki/Black_sheep"
        )
    ])
    
    return entities


def populate_knowledge_graph():
    """Populate the knowledge graph with sample data."""
    logger.info("Populating knowledge graph with sample UK cultural data...")
    
    try:
        # Initialize components
        kg = KnowledgeGraphDB()
        vs = VectorStore()
        
        # Get sample entities
        entities = create_sample_entities()
        logger.info(f"Created {len(entities)} sample entities")
        
        # Add entities to knowledge graph and vector store
        success_count = 0
        for entity in entities:
            try:
                # Add to knowledge graph
                if kg.create_entity(entity):
                    # Add to vector store
                    vs.add_entity(entity)
                    success_count += 1
                    logger.info(f"Added entity: {entity.entity_name} ({entity.entity_type.value})")
                else:
                    logger.warning(f"Failed to add entity: {entity.entity_name}")
                    
            except Exception as e:
                logger.error(f"Error adding entity {entity.entity_name}: {e}")
        
        logger.info(f"Successfully added {success_count}/{len(entities)} entities")
        
        # Get final statistics
        kg_stats = kg.get_statistics()
        vs_stats = vs.get_statistics()
        
        logger.info(f"Knowledge Graph stats: {kg_stats}")
        logger.info(f"Vector Store stats: {vs_stats}")
        
        kg.close()
        
        return success_count == len(entities)
        
    except Exception as e:
        logger.error(f"Error populating knowledge graph: {e}")
        return False


def main():
    """Main function."""
    logger.info("Starting knowledge graph population...")
    
    if populate_knowledge_graph():
        logger.info("🎉 Knowledge graph populated successfully!")
        return 0
    else:
        logger.error("❌ Knowledge graph population failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)