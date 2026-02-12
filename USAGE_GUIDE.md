# Usage Guide

This guide covers the end-to-end workflows for both standard GPU fine-tuning and ultra-efficient BitNet CPU inference.

## 1. Data Preparation

Regardless of the model type, place your training text files (`.txt`) in the `books/` folder. The system will automatically process all files in this directory.

```
books/
├── mechanics_guide.txt
├── robotics_advanced.txt
└── biology_systems.txt
```

---

## 2. Standard Model Workflow (GPU)

Use this workflow to fine-tune high-quality models like Phi-3, Llama 3, or Mistral and deploy them via Ollama.

### Step 1: Fine-Tune the Model
Run the interactive training script:
```bash
python train.py
```
1. **Select a Model**: Choose from your local Ollama models.
2. **Set Context**: Enter the maximum sequence length (e.g., 2048).
3. **Automated Data Acquisition**: By default, the system will fetch specialized datasets (Science/Coding/LEAP71) as defined in `config.py`. See [TRAINING_ADVANCED.md](TRAINING_ADVANCED.md) for details.
4. **Wait**: Training typically takes 2-6 hours on a modern GPU.

### Step 2: Convert to GGUF
Once training finishes, convert the output to the GGUF format:
```bash
python convert_to_gguf.py
```

### Step 3: Import to Ollama
Create your custom Aurora model in Ollama:
```bash
python import_to_ollama.py
```

### Step 4: Run and Test
Test your model via the CLI:
```bash
ollama run aurora
```
Or use the test script:
```bash
python test_aurora.py
```

---

## 3. BitNet Model Workflow (CPU)

Use this workflow to run 1-bit LLMs that are highly optimized for CPU inference with massive memory and energy savings.

### Step 1: Environment Setup
Ensure `bitnet.cpp` is ready (see [INSTALLATION.md](INSTALLATION.md)):
```bash
python test_bitnet.py --setup-only
```

### Step 2: Download a BitNet Model
Download an official pre-trained 1.58-bit model:
```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
model_dir = manager.download_bitnet_model('bitnet-2b') # Downloads microsoft/BitNet-b1.58-2B-4T
```

### Step 3: Prepare for Inference
Convert the downloaded model to the optimized BitNet GGUF format:
```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
# This will run model conversion and build binaries if needed
gguf_path = manager.setup_model_for_inference('bitnet_models/BitNet-b1.58-2B-4T')
```

### Step 4: Run Inference
Perform fast CPU inference:
```python
from bitnet_utils import BitNetManager

manager = BitNetManager()
response = manager.run_inference(
    model_path='bitnet_models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf',
    prompt="What are the laws of robotics?",
    n_predict=256,
    conversation_mode=True
)
print(response)
```

---

## 4. Configuration

You can customize the project by editing `config.py`:

- **LoRA Settings**: Adjust `LORA_R` and `LORA_ALPHA` for different fine-tuning intensities.
- **BitNet Settings**:
  - `BITNET_QUANT_TYPE`: Set to `"i2_s"` (optimized) or `"tl1"` (low memory).
  - `BITNET_THREADS`: Set number of CPU cores to use.
- **System Prompt**: Edit `SYSTEM_PROMPT` to change Aurora's core personality and knowledge base.

---

## 5. Troubleshooting

- **Memory Issues**: Reduce `BATCH_SIZE` or `MAX_SEQ_LENGTH` in `config.py`.
- **BitNet Build Failures**: If `setup_model_for_inference` fails to compile, see [BITNET_ISSUES.md](BITNET_ISSUES.md) for manual GCC build instructions.
- **Slow Inference**: Ensure `BITNET_THREADS` is set to the number of physical cores on your CPU.
