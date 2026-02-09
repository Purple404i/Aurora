"""
Test script for BitNet integration
Tests setup, download, and inference with BitNet models
"""

import os
import logging
from bitnet_utils import BitNetManager, setup_bitnet_environment
from config import BITNET_AVAILABLE_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bitnet_setup():
    """Test BitNet environment setup."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: BitNet Environment Setup")
    logger.info("="*60)
    
    success = setup_bitnet_environment()
    
    if success:
        logger.info("✓ BitNet setup test PASSED")
    else:
        logger.error("✗ BitNet setup test FAILED")
    
    return success


def test_model_download():
    """Test downloading a BitNet model."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Model Download")
    logger.info("="*60)
    
    try:
        manager = BitNetManager()
        
        # Test with the smallest model
        model_name = 'bitnet-b1.58-2B-4T'
        logger.info(f"\nDownloading test model: {model_name}")
        logger.info("This may take a few minutes...")
        
        model_dir = manager.download_bitnet_model(model_name)
        
        if os.path.exists(model_dir):
            logger.info(f"✓ Model download test PASSED")
            logger.info(f"  Model location: {model_dir}")
            return True, model_dir
        else:
            logger.error("✗ Model download test FAILED")
            return False, None
            
    except Exception as e:
        logger.error(f"✗ Model download test FAILED: {str(e)}")
        return False, None


def test_model_setup(model_dir: str):
    """Test setting up model for inference."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Model Setup for Inference")
    logger.info("="*60)
    
    try:
        manager = BitNetManager()
        
        logger.info("\nConverting model to GGUF format...")
        gguf_path = manager.setup_model_for_inference(model_dir)
        
        if os.path.exists(gguf_path):
            logger.info(f"✓ Model setup test PASSED")
            logger.info(f"  GGUF model: {gguf_path}")
            return True, gguf_path
        else:
            logger.error("✗ Model setup test FAILED")
            return False, None
            
    except Exception as e:
        logger.error(f"✗ Model setup test FAILED: {str(e)}")
        return False, None


def test_inference(gguf_path: str):
    """Test running inference with BitNet model."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Inference")
    logger.info("="*60)
    
    try:
        manager = BitNetManager()
        
        test_prompt = "What is artificial intelligence?"
        logger.info(f"\nPrompt: {test_prompt}")
        logger.info("\nGenerating response...")
        
        response = manager.run_inference(
            model_path=gguf_path,
            prompt=test_prompt,
            n_predict=100,
            temperature=0.7,
            conversation_mode=True
        )
        
        logger.info(f"\n{'-'*60}")
        logger.info("Response:")
        logger.info(f"{'-'*60}")
        logger.info(response)
        logger.info(f"{'-'*60}")
        
        if response and len(response.strip()) > 0:
            logger.info("✓ Inference test PASSED")
            return True
        else:
            logger.error("✗ Inference test FAILED (empty response)")
            return False
            
    except Exception as e:
        logger.error(f"✗ Inference test FAILED: {str(e)}")
        return False


def test_benchmark(gguf_path: str):
    """Test benchmarking BitNet model."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Benchmark")
    logger.info("="*60)
    
    try:
        manager = BitNetManager()
        
        logger.info("\nRunning benchmark (this may take a minute)...")
        results = manager.benchmark_model(
            model_path=gguf_path,
            n_predict=128,
            prompt_length=512
        )
        
        logger.info("✓ Benchmark test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ Benchmark test FAILED: {str(e)}")
        return False


def run_all_tests(skip_download: bool = False):
    """
    Run all BitNet tests.
    
    Args:
        skip_download: Skip download test if model already exists
    """
    logger.info("\n" + "="*60)
    logger.info("BITNET INTEGRATION TESTS")
    logger.info("="*60)
    
    results = {}
    
    # Test 1: Setup
    results['setup'] = test_bitnet_setup()
    
    if not results['setup']:
        logger.error("\n❌ Setup failed, cannot continue with other tests")
        return results
    
    # Test 2: Download
    model_dir = None
    if not skip_download:
        success, model_dir = test_model_download()
        results['download'] = success
    else:
        logger.info("\n⏭️  Skipping download test (use existing model)")
        # Try to find existing model
        manager = BitNetManager()
        model_name = 'bitnet-b1.58-2B-4T'
        model_dir = os.path.join(manager.models_dir, model_name)
        if os.path.exists(model_dir):
            results['download'] = True
        else:
            logger.error(f"Model not found at: {model_dir}")
            results['download'] = False
    
    if not results['download']:
        logger.error("\n❌ Model download failed or not found, cannot continue")
        return results
    
    # Test 3: Setup for inference
    success, gguf_path = test_model_setup(model_dir)
    results['setup_inference'] = success
    
    if not results['setup_inference']:
        logger.error("\n❌ Model setup failed, cannot continue")
        return results
    
    # Test 4: Inference
    results['inference'] = test_inference(gguf_path)
    
    # Test 5: Benchmark (optional)
    try:
        results['benchmark'] = test_benchmark(gguf_path)
    except Exception as e:
        logger.warning(f"Benchmark test skipped: {str(e)}")
        results['benchmark'] = None
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⏭️  SKIPPED"
        logger.info(f"{test_name:20s}: {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    logger.info("="*60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("="*60)
    
    return results


def interactive_test():
    """Interactive testing interface."""
    logger.info("\n" + "="*60)
    logger.info("BITNET INTERACTIVE TEST")
    logger.info("="*60)
    
    print("\nThis will test BitNet integration with your system.")
    print("\nOptions:")
    print("  1. Run all tests (including download)")
    print("  2. Run all tests (skip download if model exists)")
    print("  3. Setup only")
    print("  4. List available models")
    print("  5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        run_all_tests(skip_download=False)
    elif choice == '2':
        run_all_tests(skip_download=True)
    elif choice == '3':
        test_bitnet_setup()
    elif choice == '4':
        manager = BitNetManager()
        manager.list_available_models()
    elif choice == '5':
        logger.info("Exiting...")
    else:
        logger.error("Invalid choice")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            run_all_tests(skip_download=False)
        elif sys.argv[1] == '--skip-download':
            run_all_tests(skip_download=True)
        elif sys.argv[1] == '--setup-only':
            test_bitnet_setup()
        else:
            print("Usage:")
            print("  python test_bitnet.py              # Interactive mode")
            print("  python test_bitnet.py --all        # Run all tests")
            print("  python test_bitnet.py --skip-download  # Skip download")
            print("  python test_bitnet.py --setup-only # Setup only")
    else:
        interactive_test()