# Aurora: Advanced Technical Assistant

Aurora is a specialized AI assistant designed for robotics engineering, physics, and biological sciences. This repository provides a complete framework for fine-tuning state-of-the-art LLMs on custom data and deploying them efficiently.

## 🌟 Key Features

- **Dual Workflow Support**: Fine-tune high-quality standard models (GPU) or run ultra-efficient BitNet 1-bit models (CPU).
- **Interactive Fine-Tuning**: Automatically maps local Ollama models to Unsloth for efficient LoRA training.
- **CPU Optimized**: Integrated with `bitnet.cpp` for 2-6x faster inference on standard hardware.
- **Flexible Deployment**: Direct export to Ollama or GGUF format.

---

## 🚀 Quick Start

### 1. Installation
Follow the [Installation Guide](INSTALLATION.md) to set up your environment.

### 2. Choose Your Path

| I want to... | Hardware | Guide |
| :--- | :--- | :--- |
| **Fine-tune Llama 3 / Phi-3** | NVIDIA GPU (16GB+ VRAM) | [Usage Guide (Standard)](USAGE_GUIDE.md#2-standard-model-workflow-gpu) |
| **Advanced Training (Science/Code)** | NVIDIA GPU (24GB+ VRAM) | [Advanced Training Guide](TRAINING_ADVANCED.md) |
| **Run 1-bit LLMs on CPU** | Modern CPU (ARM/x86) | [Usage Guide (BitNet)](USAGE_GUIDE.md#3-bitnet-model-workflow-cpu) |

---

## 🛠 Project Structure

- `train.py`: Main interactive training script.
- `bitnet_utils.py`: Utilities for managing BitNet models and inference.
- `config.py`: Central configuration for all workflows.
- `data_preparation.py`: Advanced data loading and chat template formatting.
- `books/`: Directory for your custom training data (.txt files).

---

## ❓ Troubleshooting & Support

- **Common build issues**: See [BITNET_ISSUES.md](BITNET_ISSUES.md) for solutions to compiler and header problems.
- **Memory/Quality**: Refer to the troubleshooting sections in the [Usage Guide](USAGE_GUIDE.md#5-troubleshooting).

## 📄 License

This project incorporates:
- [BitNet](https://github.com/microsoft/BitNet) (MIT License)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT License)
- [Unsloth](https://github.com/unslothai/unsloth) (Apache 2.0)
