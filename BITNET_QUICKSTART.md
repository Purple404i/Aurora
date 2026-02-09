# BitNet Quick Start Guide

Get started with BitNet 1-bit LLMs in Aurora in under 10 minutes!

## What is BitNet?

BitNet is Microsoft's revolutionary 1-bit Large Language Model that uses ternary weights (-1, 0, +1) instead of standard 32-bit or 16-bit floating-point numbers. This enables:

- **16x smaller model size**: A 7B model goes from ~14GB to ~0.8GB
- **2-6x faster CPU inference**: Optimized for modern CPUs (ARM & x86)
- **55-82% energy savings**: Much more efficient than standard models
- **No GPU required**: Run massive models on regular computers

## Quick Setup (5 minutes)

### Step 1: Install Prerequisites

**Ubuntu/Debian:**
```bash
# Install Clang/LLVM
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# Install CMake and Git
sudo apt install cmake git
```

**Windows:**
- Install Visual Studio 2022
- Enable: Desktop development with C++, CMake tools, Clang compiler

**macOS:**
```bash
brew install cmake llvm git
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Setup BitNet Framework
```bash
python test_bitnet.py --setup-only
```

This will clone and setup bitnet.cpp (~5 minutes).

## Quick Test (2 minutes)

Run a complete test of BitNet:

```bash
python test_bitnet.py --all
```

This will:
1. ✓ Setup bitnet.cpp
2. ✓ Download bitnet-b1.58-2B-4T model (~2GB)
3. ✓ Convert to GGUF format
4. ✓ Run inference test
5. ✓ Run performance benchmark

## Usage Examples

### Example 1: Simple Inference

```python
from bitnet_utils import BitNetManager

# Initialize manager
manager = BitNetManager()

# Download model (first time only)
model_dir = manager.download_bitnet_model('bitnet-b1.58-2B-4T')

# Setup for inference (first time only)
gguf_path = manager.setup_model_for_inference(model_dir)

# Run inference
response = manager.run_inference(
    model_path=gguf_path,
    prompt="Explain quantum computing in simple terms.",
    n_predict=200,
    temperature=0.7
)

print(response)
```

### Example 2: Conversation Mode

```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
gguf_path = "bitnet_models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Use conversation mode for instruct models
response = manager.run_inference(
    model_path=gguf_path,
    prompt="What are the benefits of using 1-bit LLMs?",
    n_predict=300,
    temperature=0.7,
    conversation_mode=True  # Enables chat mode
)

print(response)
```

### Example 3: Benchmark Your System

```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
gguf_path = "bitnet_models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Benchmark performance
results = manager.benchmark_model(
    model_path=gguf_path,
    n_predict=128,      # Generate 128 tokens
    prompt_length=512,  # From 512 token prompt
    threads=8           # Use 8 CPU threads
)

print(results)
```

### Example 4: Using Command Line

```bash
cd bitnet.cpp

# Single query
python run_inference.py \
    -m ../bitnet_models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "Write a haiku about AI" \
    -n 100 \
    -temp 0.8

# Interactive chat mode
python run_inference.py \
    -m ../bitnet_models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "You are a helpful AI assistant" \
    -cnv
```

## Available BitNet Models

| Model Name | Size | Parameters | Best For |
|------------|------|------------|----------|
| bitnet-b1.58-2B-4T | ~2GB | 2B | General use, fastest |
| bitnet-b1.58-3B | ~3GB | 3B | Better quality |
| bitnet-b1.58-large | ~7GB | 7B | High quality |
| llama3-8B-1.58bit | ~8GB | 8B | Llama3-based |
| falcon3-7B-1.58bit | ~7GB | 7B | Falcon3-based |

Download any model:
```python
manager.download_bitnet_model('llama3-8B-1.58bit')
```

## Configuration Options

Edit `config.py` to customize BitNet settings:

```python
# Quantization type (i2_s recommended, tl1 for lower memory)
BITNET_QUANT_TYPE = "i2_s"

# Number of CPU threads (None = auto-detect)
BITNET_THREADS = 8

# Context window size
BITNET_CTX_SIZE = 2048

# Inference settings
BITNET_INFERENCE_SETTINGS = {
    'temperature': 0.7,      # Randomness (0.0-1.0)
    'n_predict': 512,        # Max tokens to generate
    'conversation_mode': True # Enable chat mode
}
```

## Performance Tips

### For Best Speed:
1. Use `i2_s` quantization (default)
2. Set `BITNET_THREADS` to your CPU core count
3. Use smaller models for interactive use
4. Enable CPU optimizations in BIOS (Turbo Boost, etc.)

### For Lowest Memory:
1. Use `tl1` quantization instead of `i2_s`
2. Use smaller context size (`BITNET_CTX_SIZE = 1024`)
3. Choose smaller models (2B-3B parameters)

### For Best Quality:
1. Use larger models (7B-8B parameters)
2. Keep `temperature` around 0.7
3. Use `i2_s` quantization
4. Increase context size if needed

## Troubleshooting

### Setup Issues

**Error: "clang not found"**
```bash
# Ubuntu/Debian
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# macOS
brew install llvm

# Windows
Use Visual Studio Developer Command Prompt
```

**Error: "CMake not found"**
```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# Windows
Install via Visual Studio installer
```

### Runtime Issues

**Slow inference**
- Increase thread count: `BITNET_THREADS = 16`
- Close other applications
- Use smaller model
- Try `tl1` quantization

**High memory usage**
- Use `tl1` quantization instead of `i2_s`
- Reduce context size: `BITNET_CTX_SIZE = 1024`
- Use smaller model

**Model download fails**
```python
# Manual download
import subprocess
subprocess.run([
    "huggingface-cli", "download",
    "microsoft/bitnet-b1.58-2B-4T",
    "--local-dir", "./bitnet_models/bitnet-b1.58-2B-4T"
])
```

## Next Steps

1. **Fine-tune on your data**: While BitNet is optimized for inference, you can still fine-tune the base model before converting
2. **Deploy to production**: BitNet models are perfect for edge deployment
3. **Try different models**: Each model has different strengths
4. **Optimize for your hardware**: Experiment with thread count and quantization

## Resources

- **BitNet Paper**: https://arxiv.org/abs/2402.17764
- **BitNet GitHub**: https://github.com/microsoft/BitNet
- **Model Hub**: https://huggingface.co/collections/1bitLLM
- **Aurora Docs**: See README_BITNET.md for full documentation

## Common Use Cases

### Edge Deployment
```python
# Perfect for Raspberry Pi, embedded systems, edge servers
# No GPU required, minimal power consumption
manager = BitNetManager()
response = manager.run_inference(
    model_path=gguf_path,
    prompt="Your query",
    threads=4  # Adjust to your CPU cores
)
```

### Local Development
```python
# Great for local testing without cloud costs
# Fast iteration on CPU
manager = BitNetManager()
response = manager.run_inference(
    model_path=gguf_path,
    prompt="Your query",
    conversation_mode=True
)
```

### Batch Processing
```python
# Process many queries efficiently
prompts = ["Query 1", "Query 2", "Query 3"]
for prompt in prompts:
    response = manager.run_inference(
        model_path=gguf_path,
        prompt=prompt,
        n_predict=100
    )
    print(f"{prompt}: {response}\n")
```

## FAQ

**Q: Can I fine-tune BitNet models?**
A: BitNet models are primarily designed for efficient inference. Fine-tuning is complex due to the 1-bit quantization. Consider fine-tuning a standard model first, then converting.

**Q: How does BitNet compare to standard quantization?**
A: BitNet uses 1.58-bit (ternary) weights trained from scratch, while standard quantization reduces precision after training. BitNet is more efficient but requires specialized training.

**Q: Can I use BitNet on GPU?**
A: BitNet is optimized for CPU. GPU support is planned but CPU is the primary target and often faster due to optimized kernels.

**Q: Which model should I start with?**
A: Start with `bitnet-b1.58-2B-4T` - it's small, fast, and good quality for testing.

## Support

- Open an issue on GitHub
- Check BitNet documentation
- See full Aurora documentation in README_BITNET.md

---

**Ready to start?** Run `python test_bitnet.py --all` now! 🚀