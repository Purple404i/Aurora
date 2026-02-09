"""
Utilities for working with BitNet models
Handles setup, conversion, and inference for 1-bit LLMs
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, List
import json

from config import (
    BITNET_CPP_DIR,
    BITNET_MODELS_DIR,
    BITNET_QUANT_TYPE,
    BITNET_USE_PRETUNED,
    BITNET_THREADS,
    BITNET_CTX_SIZE,
    BITNET_INFERENCE_SETTINGS,
    BITNET_AVAILABLE_MODELS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BitNetManager:
    """
    Manager class for BitNet models and bitnet.cpp framework.
    """
    
    def __init__(self, cpp_dir: str = BITNET_CPP_DIR, models_dir: str = BITNET_MODELS_DIR):
        """
        Initialize BitNet manager.
        
        Args:
            cpp_dir: Directory for bitnet.cpp repository
            models_dir: Directory for BitNet models
        """
        self.cpp_dir = cpp_dir
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
    
    def setup_bitnet_cpp(self) -> bool:
        """
        Clone and build bitnet.cpp if not already set up.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info("Setting up bitnet.cpp...")
            
            # Clone repository if not exists
            if not os.path.exists(self.cpp_dir):
                logger.info("Cloning bitnet.cpp repository...")
                subprocess.run([
                    "git", "clone", "--recursive",
                    "https://github.com/microsoft/BitNet.git",
                    self.cpp_dir
                ], check=True)
            else:
                logger.info("bitnet.cpp repository already exists")
            
            # Check for required dependencies
            self._check_dependencies()
            
            logger.info("✓ bitnet.cpp setup complete")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup bitnet.cpp: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during setup: {str(e)}")
            return False
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            # Check Python version
            import sys
            if sys.version_info < (3, 9):
                logger.warning("Python 3.9+ recommended for bitnet.cpp")
            
            # Check for required packages
            required_packages = ['huggingface_hub', 'torch', 'transformers']
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.warning(f"Missing packages: {', '.join(missing_packages)}")
                logger.info("Install with: pip install " + " ".join(missing_packages))
            
        except Exception as e:
            logger.warning(f"Could not check dependencies: {str(e)}")
    
    def download_bitnet_model(self, model_name: str, hf_repo: Optional[str] = None) -> str:
        """
        Download a BitNet model from HuggingFace.
        
        Args:
            model_name: Name of the model
            hf_repo: HuggingFace repository (optional, uses BITNET_AVAILABLE_MODELS if not provided)
            
        Returns:
            Path to downloaded model
        """
        try:
            # Get HuggingFace repo
            if hf_repo is None:
                if model_name in BITNET_AVAILABLE_MODELS:
                    hf_repo = BITNET_AVAILABLE_MODELS[model_name]
                else:
                    hf_repo = model_name
            
            logger.info(f"Downloading BitNet model: {hf_repo}")
            
            model_dir = os.path.join(self.models_dir, model_name)
            
            # Use huggingface-cli to download
            subprocess.run([
                "huggingface-cli", "download",
                hf_repo,
                "--local-dir", model_dir
            ], check=True)
            
            logger.info(f"✓ Model downloaded to: {model_dir}")
            return model_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def setup_model_for_inference(
        self,
        model_dir: str,
        quant_type: str = BITNET_QUANT_TYPE,
        use_pretuned: bool = BITNET_USE_PRETUNED
    ) -> str:
        """
        Setup a BitNet model for inference (convert to GGUF if needed).
        
        Args:
            model_dir: Directory containing the model
            quant_type: Quantization type (i2_s or tl1)
            use_pretuned: Whether to use pre-tuned kernels
            
        Returns:
            Path to GGUF model file
        """
        try:
            logger.info(f"Setting up model for inference: {model_dir}")
            
            # Check if GGUF file already exists
            gguf_pattern = f"ggml-model-{quant_type}.gguf"
            gguf_path = os.path.join(model_dir, gguf_pattern)
            
            if os.path.exists(gguf_path):
                logger.info(f"✓ GGUF model already exists: {gguf_path}")
                return gguf_path
            
            # Run setup_env.py from bitnet.cpp
            setup_script = os.path.join(self.cpp_dir, "setup_env.py")
            
            cmd = [
                "python", setup_script,
                "-md", model_dir,
                "-q", quant_type
            ]
            
            if use_pretuned:
                cmd.append("--use-pretuned")
            
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.cpp_dir)
            
            if not os.path.exists(gguf_path):
                raise FileNotFoundError(f"GGUF file not created: {gguf_path}")
            
            logger.info(f"✓ Model ready for inference: {gguf_path}")
            return gguf_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup model: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def run_inference(
        self,
        model_path: str,
        prompt: str,
        n_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        threads: Optional[int] = None,
        ctx_size: Optional[int] = None,
        conversation_mode: bool = True
    ) -> str:
        """
        Run inference using bitnet.cpp.
        
        Args:
            model_path: Path to GGUF model file
            prompt: Input prompt
            n_predict: Number of tokens to predict
            temperature: Sampling temperature
            threads: Number of threads to use
            ctx_size: Context size
            conversation_mode: Enable conversation mode for instruct models
            
        Returns:
            Generated text
        """
        try:
            # Use config defaults if not provided
            n_predict = n_predict or BITNET_INFERENCE_SETTINGS['n_predict']
            temperature = temperature or BITNET_INFERENCE_SETTINGS['temperature']
            threads = threads or BITNET_THREADS
            ctx_size = ctx_size or BITNET_CTX_SIZE
            
            # Build command
            inference_script = os.path.join(self.cpp_dir, "run_inference.py")
            
            cmd = [
                "python", inference_script,
                "-m", model_path,
                "-p", prompt,
                "-n", str(n_predict),
                "-temp", str(temperature),
                "-c", str(ctx_size)
            ]
            
            if threads:
                cmd.extend(["-t", str(threads)])
            
            if conversation_mode:
                cmd.append("-cnv")
            
            logger.info(f"Running inference...")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.cpp_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Inference failed: {str(e)}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def benchmark_model(
        self,
        model_path: str,
        n_predict: int = 128,
        prompt_length: int = 512,
        threads: Optional[int] = None
    ) -> Dict:
        """
        Benchmark a BitNet model.
        
        Args:
            model_path: Path to GGUF model file
            n_predict: Number of tokens to generate
            prompt_length: Length of prompt in tokens
            threads: Number of threads to use
            
        Returns:
            Benchmark results dictionary
        """
        try:
            logger.info(f"Benchmarking model: {model_path}")
            
            benchmark_script = os.path.join(self.cpp_dir, "utils", "e2e_benchmark.py")
            
            cmd = [
                "python", benchmark_script,
                "-m", model_path,
                "-n", str(n_predict),
                "-p", str(prompt_length)
            ]
            
            if threads:
                cmd.extend(["-t", str(threads)])
            
            result = subprocess.run(
                cmd,
                cwd=self.cpp_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Benchmark results:\n{result.stdout}")
            
            # Parse results (basic parsing, adjust as needed)
            return {
                'output': result.stdout,
                'model': model_path,
                'n_predict': n_predict,
                'prompt_length': prompt_length
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark failed: {str(e)}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def convert_hf_to_bitnet(
        self,
        hf_model_dir: str,
        output_name: str,
        quant_type: str = BITNET_QUANT_TYPE
    ) -> str:
        """
        Convert a HuggingFace model to BitNet format.
        
        Args:
            hf_model_dir: Directory containing HuggingFace model
            output_name: Name for output model
            quant_type: Quantization type
            
        Returns:
            Path to converted GGUF model
        """
        try:
            logger.info(f"Converting HuggingFace model to BitNet format...")
            logger.info(f"Source: {hf_model_dir}")
            
            # Use the helper conversion script if available
            convert_script = os.path.join(self.cpp_dir, "utils", "convert-helper-bitnet.py")
            
            if not os.path.exists(convert_script):
                logger.warning("Conversion helper not found, using setup_env.py instead")
                return self.setup_model_for_inference(hf_model_dir, quant_type)
            
            output_dir = os.path.join(self.models_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                "python", convert_script,
                hf_model_dir
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.cpp_dir)
            
            # Setup for inference
            return self.setup_model_for_inference(output_dir, quant_type)
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            raise
    
    def list_available_models(self) -> List[str]:
        """
        List available BitNet models.
        
        Returns:
            List of model names
        """
        logger.info("Available BitNet models:")
        for name, repo in BITNET_AVAILABLE_MODELS.items():
            logger.info(f"  - {name}: {repo}")
        return list(BITNET_AVAILABLE_MODELS.keys())


def setup_bitnet_environment():
    """
    Complete setup for BitNet environment.
    """
    logger.info("="*60)
    logger.info("BITNET ENVIRONMENT SETUP")
    logger.info("="*60)
    
    manager = BitNetManager()
    
    # Setup bitnet.cpp
    if not manager.setup_bitnet_cpp():
        logger.error("Failed to setup bitnet.cpp")
        return False
    
    # List available models
    manager.list_available_models()
    
    logger.info("\n" + "="*60)
    logger.info("SETUP COMPLETE")
    logger.info("="*60)
    logger.info("\nTo download a model, use:")
    logger.info("  from bitnet_utils import BitNetManager")
    logger.info("  manager = BitNetManager()")
    logger.info("  manager.download_bitnet_model('bitnet-b1.58-2B-4T')")
    
    return True


if __name__ == "__main__":
    # Test the setup
    setup_bitnet_environment()