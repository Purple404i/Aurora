"""
Import the Aurora GGUF model into Ollama
"""

import os
import subprocess
from config import OUTPUT_MODEL_NAME, SYSTEM_PROMPT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_modelfile():
    """
    Create a Modelfile for Ollama
    """
    gguf_file = f"./aurora_gguf/{OUTPUT_MODEL_NAME}.gguf"
    
    if not os.path.exists(gguf_file):
        logger.error(f"GGUF file not found: {gguf_file}")
        logger.error("Please run convert_to_gguf.py first")
        return None
    
    modelfile_content = f"""FROM {gguf_file}

# Temperature (creativity)
PARAMETER temperature 0.7

# Context window
PARAMETER num_ctx 4096

# System prompt
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"
"""
    
    modelfile_path = "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    logger.info(f"Modelfile created: {modelfile_path}")
    return modelfile_path

def import_to_ollama():
    """
    Import Aurora model into Ollama
    """
    logger.info("="*60)
    logger.info("IMPORTING AURORA TO OLLAMA")
    logger.info("="*60)
    
    # Create Modelfile
    modelfile = create_modelfile()
    
    if not modelfile:
        return
    
    # Import to Ollama
    logger.info(f"\nImporting {OUTPUT_MODEL_NAME} to Ollama...")
    
    try:
        subprocess.run([
            "ollama", "create", OUTPUT_MODEL_NAME, "-f", modelfile
        ], check=True)
        
        logger.info(f"\n{'='*60}")
        logger.info("IMPORT COMPLETE!")
        logger.info("="*60)
        logger.info(f"\nAurora is now available in Ollama!")
        logger.info(f"\nTo use Aurora, run:")
        logger.info(f"  ollama run {OUTPUT_MODEL_NAME}")
        logger.info(f"\nTo list all models:")
        logger.info(f"  ollama list")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to import to Ollama: {str(e)}")
        logger.error("Make sure Ollama is installed and running")
        raise

if __name__ == "__main__":
    try:
        import_to_ollama()
    except Exception as e:
        logger.error(f"Import failed: {str(e)}", exc_info=True)
        raise