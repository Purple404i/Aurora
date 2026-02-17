# Aurora: Advanced Technical Assistant

Aurora is an interdisciplinary AI assistant specialized in **Robotics Engineering**, **Physics**, **Mathematics**, **Circuit Design**, and **Biological Sciences**. This repository provides the complete framework to acquire specialized knowledge, fine-tune models, and deploy them as **Aurora**.

## 🌟 Key Features

- **Automated Data Acquisition**: Fetch technical datasets from Hugging Face and clone professional repos (LEAP71, RobotCEM).
- **Dual Inference Support**: Deploy high-quality models in Ollama (GPU) or ultra-efficient 1-bit models in bitnet.cpp (CPU).
- **Interactive Training**: Unified fine-tuning pipeline for standard and BitNet models using Unsloth.
- **PicoGK/RobotCEM Ready**: Specialized training for voxel-based computational engineering.

---

## ⚡ Workflow at a Glance

To build your own **Aurora**, run these scripts in order:

1.  **`python data_preparation.py`**: Gathers all technical knowledge into `books/`.
2.  **`python train.py`**: Performs the interactive fine-tuning.
3.  **`python convert_to_gguf.py`**: Prepares the model for Ollama.
4.  **`python import_to_ollama.py`**: Creates the **"aurora"** model.

---

## 🚀 Getting Started

### 1. Installation
Follow the [Installation Guide](INSTALLATION.md) to set up your environment (GPU for training, CPU for BitNet).

### 2. Detailed Guides

| Guide | Description |
| :--- | :--- |
| **[Usage Guide](USAGE_GUIDE.md)** | Step-by-step instructions and "Order of Operations". |
| **[Advanced Training](TRAINING_ADVANCED.md)** | Deep dive into specialized datasets (Math, Circuits, Mechanics). |
| **[Troubleshooting](BITNET_ISSUES.md)** | Solutions for common build and path issues. |

---

## 🛠 Project Map

- `config.py`: Central configuration for datasets, models, and prompts.
- `data_preparation.py`: The "Acquisition" engine (HF + GitHub).
- `bitnet_utils.py`: The 1-bit infrastructure manager.
- `train.py`: The fine-tuning orchestrator.
- `convert_to_gguf.py`: Export utility for standard models.
- `import_to_ollama.py`: Ollama integration script.

---

## 📄 License

This project incorporates:
- [BitNet](https://github.com/microsoft/BitNet) (MIT License)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT License)
- [Unsloth](https://github.com/unslothai/unsloth) (Apache 2.0)
