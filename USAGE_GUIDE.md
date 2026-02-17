# Usage Guide: Step-by-Step Workflow

This guide provides the exact order of execution to prepare data, train your model, and transform it into **Aurora** for use in Ollama or bitnet.cpp.

---

## 1. Data Preparation (Acquisition)
**File to run**: `data_preparation.py`

Before training, you must gather the technical knowledge Aurora needs.
```bash
python data_preparation.py
```
**What this does:**
1. **Fetches HF Datasets**: Downloads Mathematics, Circuits, Mechanics, and Science datasets from Hugging Face.
2. **Clones Technical Repos**: Clones LEAP71 (PicoGK) and RobotCEM repositories.
3. **Extracts Content**: Pulls documentation and code (`.cs`, `.py`, `.v`, `.md`) from these repos.
4. **Populates `books/`**: Saves everything as `.txt` files in the `books/` directory.

*Note: You can inspect the `books/` folder after this step to see exactly what Aurora will learn.*

---

## 2. Training Workflow: Standard Models (Llama 3, Phi-3, etc.)

Use this path if you have an NVIDIA GPU and want a high-quality model deployed in Ollama.

### Step A: Fine-Tuning
**File to run**: `train.py`
```bash
python train.py
```
**The Process:**
1. **Select Base**: Choose a model already in your Ollama library (e.g., `llama3`).
2. **Interactive Config**: The script asks for context length and automatically updates `config.py`.
3. **Training**: It uses Unsloth to fine-tune the model on the data in `books/`.
4. **Output**: The results are saved in `aurora_output/`.

### Step B: GGUF Conversion
**File to run**: `convert_to_gguf.py`
```bash
python convert_to_gguf.py
```
**What this does**: Merges the trained LoRA weights with the base model and exports it as a `.gguf` file in the `aurora_gguf/` folder. This format is required for Ollama.

### Step C: Create "Aurora" in Ollama
**File to run**: `import_to_ollama.py`
```bash
python import_to_ollama.py
```
**What this does**: Reads the GGUF file and the configuration to create a new model named **"aurora"** in your local Ollama instance.

### Step D: Run & Test
```bash
ollama run aurora
```
---

## 3. Training Workflow: BitNet Models (1-bit LLMs)

Use this path for ultra-efficient CPU inference.

### Step A: Environment Setup
**File to run**: `test_bitnet.py`
```bash
python test_bitnet.py --setup-only
```
**What this does**: Clones `bitnet.cpp` and ensures the environment is ready for 1-bit operations.

### Step B: Fine-Tuning
**File to run**: `train.py`
```bash
python train.py
```
1. **Choose BitNet**: Pick a BitNet model (e.g., `bitnet-2b`).
2. **Optimization**: `train.py` automatically disables 4-bit loading (as BitNet is 1.58-bit) and applies specialized LoRA settings.

### Step C: Setup for Inference
**Manual Step**: Use the `BitNetManager` in `bitnet_utils.py` or the provided CLI.
```python
from bitnet_utils import BitNetManager
manager = BitNetManager()
# Converts your trained model to BitNet GGUF format
manager.setup_model_for_inference('bitnet_models/BitNet-b1.58-2B-4T')
```

### Step D: Run Inference
```python
response = manager.run_inference(
    model_path='bitnet_models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf',
    prompt="Design a robotic arm joint.",
    conversation_mode=True
)
```

---

## Summary Table: Order of Operations

| Step | Goal | Standard Model (GPU) | BitNet Model (CPU) |
| :--- | :--- | :--- | :--- |
| **1** | Get Data | `data_preparation.py` | `data_preparation.py` |
| **2** | Setup Framework | N/A (Standard Pip) | `test_bitnet.py --setup-only` |
| **3** | Fine-Tune | `train.py` | `train.py` |
| **4** | Convert | `convert_to_gguf.py` | `bitnet_utils.py` (setup_model) |
| **5** | Deploy/Run | `import_to_ollama.py` | `bitnet_utils.py` (run_inference) |

---

## Configuration Management

All scripts share `config.py`. Key settings include:
- `HF_DATASETS`: List of Hugging Face repositories to fetch.
- `LEAP71_REPOS`: List of GitHub technical repos to clone.
- `SYSTEM_PROMPT`: The core personality and reasoning protocol of Aurora.
- `MAX_SEQ_LENGTH`: The context window (default 2048-8192).
