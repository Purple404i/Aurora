"""
Convert the fine-tuned Aurora model to GGUF format for Ollama
"""

import os
import subprocess
from pathlib import Path
from config import OUTPUT_DIR, OUTPUT_MODEL_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_gguf():
    """
    Convert the merged model to GGUF format using llama.cpp
    """
    logger.info("="*60)
    logger.info("CONVERTING AURORA TO GGUF FORMAT")
    logger.info("="*60)
    
    merged_model_dir = f"{OUTPUT_DIR}_merged"
    gguf_output_dir = "./aurora_gguf"
    
    if not os.path.exists(merged_model_dir):
        logger.error(f"Merged model not found at: {merged_model_dir}")
        logger.error("Please run train.py first to create the merged model.")
        return
    
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    # Clone llama.cpp if not exists
    llama_cpp_dir = os.path.abspath("./llama.cpp")
    if not os.path.exists(llama_cpp_dir):
        logger.info("Cloning llama.cpp repository...")
        subprocess.run([
            "git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir
        ], check=True)
        
        logger.info("Building llama.cpp...")
        try:
            # Try building with make
            subprocess.run(["make", "-j"], cwd=llama_cpp_dir, check=True)
        except Exception as e:
            logger.warning(f"Build with make failed: {e}. Trying with CMake...")
            build_dir = os.path.join(llama_cpp_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
            subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)

    # Install requirements for conversion
    logger.info("Installing conversion requirements...")
    subprocess.run([
        "pip", "install", "-r", os.path.join(llama_cpp_dir, "requirements.txt")
    ], check=True)
    
    # Convert to GGUF
    logger.info(f"\nConverting {merged_model_dir} to GGUF...")
    
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    output_file = os.path.abspath(os.path.join(gguf_output_dir, f"{OUTPUT_MODEL_NAME}.gguf"))
    
    subprocess.run([
        "python", convert_script,
        merged_model_dir,
        "--outfile", output_file,
        "--outtype", "f16"
    ], check=True)
    
    logger.info(f"\n{'='*60}")
    logger.info("CONVERSION COMPLETE!")
    logger.info("="*60)
    logger.info(f"GGUF model saved to: {output_file}")
    logger.info(f"\nNext step: Run import_to_ollama.py to import into Ollama")

if __name__ == "__main__":
    try:
        convert_to_gguf()
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}", exc_info=True)
        raise