# Aurora Fine-Tuning Project

Fine-tune language models on custom book data to create Aurora. Now supports **BitNet 1-bit LLMs** for ultra-efficient CPU inference!

## Supported Models

- **Phi-3-mini** (3.8B parameters, efficient)
- **Llama** (7B-70B parameters)
- **Mistral** (7B parameters, efficient)
- **BitNet** (1-bit LLMs, ultra-efficient, CPU-optimized) ⭐ NEW!

## Prerequisites

1. **Python 3.9+** installed
2. **CUDA-capable GPU** (recommended: 16GB+ VRAM) - for standard models
3. **CPU** (for BitNet models - no GPU required!)
4. **Ollama** installed (for final deployment of standard models)

### BitNet-Specific Requirements

For BitNet models, you'll also need:
- **CMake 3.22+**
- **Clang 18+** (for optimal performance)
- **Git** (for cloning bitnet.cpp)

Install on Ubuntu/Debian:
```bash
# Install LLVM/Clang
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# Install other dependencies
sudo apt install cmake git
```

Install on Windows:
- Visual Studio 2022 with:
  - Desktop development with C++
  - C++ CMake tools
  - Git for Windows
  - C++ Clang Compiler
  - MS-Build Support for LLVM

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup BitNet (Optional)

If you want to use BitNet models:

```bash
python test_bitnet.py --setup-only
```

This will clone and setup the bitnet.cpp framework.

### 3. Prepare Your Data

Place all your training text files (.txt) in the `books/` folder:
```
books/
├── book1.txt
├── book2.txt
└── book3.txt
```

## Training Process

### Standard Models (Phi-3, Llama, Mistral)

#### Step 1: Train the Model
```bash
python train.py
```

This will initiate an interactive process where you can:
- **Select a Model**: Choose from models already downloaded locally via Ollama.
- **Configure Context**: Specify the maximum sequence length (e.g., 2048, 4096) for training.
- **Automatic Optimization**: The script automatically detects the model's architecture and selects the correct target modules for LoRA fine-tuning.
- **Dynamic Formatting**: Training data is automatically formatted using the selected model's native chat template.

Training time: 2-6 hours (depending on data size and GPU)

#### Step 2: Convert to GGUF (for Ollama)
```bash
python convert_to_gguf.py
```

This converts the model to GGUF format compatible with Ollama.

#### Step 3: Import to Ollama
```bash
python import_to_ollama.py
```

This creates the Aurora model in Ollama.

#### Step 4: Test Aurora
```bash
python test_aurora.py
```

Or use Ollama directly:
```bash
ollama run aurora
```

### BitNet Models (1-bit LLMs)

BitNet models use a different workflow optimized for CPU inference:

#### Step 1: Test BitNet Setup
```bash
python test_bitnet.py --all
```

This will:
1. Setup bitnet.cpp framework
2. Download a test BitNet model (bitnet-b1.58-2B-4T)
3. Convert to GGUF format
4. Run inference test
5. Run benchmark

#### Step 2: Download Your Preferred BitNet Model

Available models:
- `bitnet-b1.58-large` - Large model
- `bitnet-b1.58-3B` - 3B parameters
- `bitnet-b1.58-2B-4T` - 2B parameters, 4T tokens (recommended)
- `llama3-8B-1.58bit` - Llama3-based 8B model
- `falcon3-1B-1.58bit` - Falcon3 1B
- `falcon3-3B-1.58bit` - Falcon3 3B
- `falcon3-7B-1.58bit` - Falcon3 7B
- `falcon3-10B-1.58bit` - Falcon3 10B

```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
model_dir = manager.download_bitnet_model('bitnet-b1.58-2B-4T')
```

#### Step 3: Setup Model for Inference
```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
gguf_path = manager.setup_model_for_inference(model_dir)
```

#### Step 4: Run Inference
```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
response = manager.run_inference(
    model_path=gguf_path,
    prompt="What is artificial intelligence?",
    n_predict=512,
    temperature=0.7
)
print(response)
```

Or use the command line:
```bash
cd bitnet.cpp
python run_inference.py \
    -m ../bitnet_models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "Your prompt here" \
    -cnv
```

## Configuration

### Standard Model Configuration

Edit `config.py` to customize:
- **Training Hyperparameters**: epochs, batch size, learning rate, etc.
- **LoRA Settings**: rank (r), alpha, and dropout.
- **System Prompt**: The core identity and instructions for Aurora.
- **Data Paths**: The folder containing your training books.

### BitNet Configuration

BitNet-specific settings in `config.py`:
- `BITNET_QUANT_TYPE`: Quantization type (i2_s or tl1)
- `BITNET_THREADS`: Number of CPU threads (None = auto)
- `BITNET_CTX_SIZE`: Context window size
- `BITNET_INFERENCE_SETTINGS`: Temperature, max tokens, etc.

## Project Structure

```
aurora/
├── books/                      # Your training data (*.txt files)
├── config.py                   # Configuration settings (includes BitNet)
├── data_preparation.py         # Data loading and preprocessing
├── train.py                    # Main training script
├── bitnet_utils.py            # BitNet utilities (NEW!)
├── test_bitnet.py             # BitNet testing script (NEW!)
├── convert_to_gguf.py         # Convert to GGUF format
├── import_to_ollama.py        # Import to Ollama
├── test_aurora.py             # Test the model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── aurora_output/             # Fine-tuned model (created during training)
├── aurora_output_merged/      # Merged model (created during training)
├── aurora_checkpoints/        # Training checkpoints
├── aurora_gguf/              # GGUF format model
├── bitnet.cpp/               # BitNet.cpp framework (NEW!)
├── bitnet_models/            # Downloaded BitNet models (NEW!)
├── logs/                     # Training logs
├── data_samples/             # Sample training data
└── llama.cpp/                # llama.cpp for conversion (auto-downloaded)
```

## BitNet Advantages

### Why Use BitNet?

1. **Extreme Efficiency**: 1.58-bit weights (ternary: -1, 0, +1) reduce memory by 16x
2. **CPU Performance**: Optimized for CPU inference, no GPU required
3. **Energy Savings**: 55-82% less energy consumption vs. standard models
4. **Speed**: 2-6x faster inference on CPU vs. standard quantized models
5. **Large Models on CPU**: Run 100B parameter models at human reading speed (5-7 tokens/sec)

### BitNet Performance

On Apple M2 Ultra (ARM):
- Speedup: 1.37x to 5.07x vs. llama.cpp
- Energy reduction: 55.4% to 70.0%

On Intel i7-13700H (x86):
- Speedup: 2.37x to 6.17x vs. llama.cpp
- Energy reduction: 71.9% to 82.2%

### When to Use BitNet

✅ **Use BitNet when:**
- Running on CPU-only systems
- Deploying on edge devices
- Need ultra-low memory usage
- Energy efficiency is critical
- Want to run large models locally

❌ **Use standard models when:**
- Have powerful GPU available
- Need absolute best quality
- Fine-tuning is primary goal (BitNet is mainly for inference)

## Monitoring Training

Training logs are saved to `logs/` folder. Monitor progress:
```bash
tail -f logs/training_*.log
```

## Troubleshooting

### Standard Models

#### Out of Memory Error
- Reduce `BATCH_SIZE` in `config.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Reduce `MAX_SEQ_LENGTH`

#### Slow Training
- Ensure GPU is being used (check with `nvidia-smi`)
- Reduce dataset size for faster iterations
- Use smaller `LORA_R` value

#### Model Quality Issues
- Increase `NUM_EPOCHS`
- Add more training data
- Adjust `LEARNING_RATE`

### BitNet Models

#### bitnet.cpp Setup Failed
- Ensure CMake 3.22+ is installed
- Ensure Clang 18+ is installed
- Check that git submodules were cloned: `cd bitnet.cpp && git submodule update --init --recursive`

#### Slow Inference
- Increase `BITNET_THREADS` in config.py
- Try different quantization type (i2_s vs. tl1)
- Use smaller model if available

#### Model Download Failed
- Check internet connection
- Verify HuggingFace access
- Try manual download: `hf download <model-name>`

## BitNet Resources

- **Paper**: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
- **Repository**: https://github.com/microsoft/BitNet
- **Models**: https://huggingface.co/collections/1bitLLM
- **Technical Report**: Available in Microsoft Research

## Notes

- The original models in Ollama remain unchanged
- Aurora is created as a separate model
- You can run multiple models side by side
- Training progress is logged to both console and log files
- BitNet models use a different inference engine (bitnet.cpp) optimized for 1-bit operations

## Support

For issues or questions, check:
- Unsloth documentation: https://github.com/unslothai/unsloth
- Ollama documentation: https://ollama.ai/
- BitNet documentation: https://github.com/microsoft/BitNet
- BitNet paper: https://arxiv.org/abs/2402.17764

## License

This project uses:
- BitNet (MIT License)
- llama.cpp (MIT License)
- Unsloth (Apache 2.0)