"""
Test the Aurora model after fine-tuning
"""

import subprocess
import json
from config import OUTPUT_MODEL_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_with_ollama():
    """
    Test Aurora using Ollama
    """
    logger.info("="*60)
    logger.info("TESTING AURORA MODEL")
    logger.info("="*60)
    
    test_prompts = [
        "Tell me about the content you were trained on.",
        "What can you help me with?",
        "Summarize the main themes from your training data.",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nTest {i}/{len(test_prompts)}")
        logger.info(f"Prompt: {prompt}")
        logger.info("-"*60)
        
        try:
            result = subprocess.run(
                ["ollama", "run", OUTPUT_MODEL_NAME, prompt],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Response:\n{result.stdout}")
            logger.info("-"*60)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running test: {e}")
            logger.error(f"Make sure Aurora is imported to Ollama")
            return
    
    logger.info("\n" + "="*60)
    logger.info("TESTING COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    test_model_with_ollama()