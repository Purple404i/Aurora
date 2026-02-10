# Installation Guide

This document provides step-by-step instructions for setting up the Aurora Fine-Tuning Project environment.

## 1. System Requirements

### For Standard Models (Phi-3, Llama, Mistral)
- **OS**: Linux (Ubuntu recommended) or Windows (WSL2 recommended)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, 4090, A100)
- **Drivers**: CUDA 12.1+

### For BitNet Models (1-bit LLMs)
- **OS**: Linux, Windows, or macOS
- **Python**: 3.9+
- **CPU**: Optimized for modern CPUs (ARM or x86_64)
- **Build Tools**:
  - **CMake**: 3.22+
  - **Compiler**: Clang 18+ (recommended) or GCC 13+

---

## 2. Basic Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/aurora.git
cd aurora
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Workflow-Specific Setup

### Standard Workflow Setup (Ollama)
If you plan to use standard models and deploy via Ollama:
1. **Install Ollama**: Follow instructions at [ollama.ai](https://ollama.ai).
2. **Download a base model**:
   ```bash
   ollama pull phi3:mini
   ```

### BitNet Workflow Setup (bitnet.cpp)
If you want to use ultra-efficient 1-bit models on CPU:

1. **Initialize bitnet.cpp**:
   ```bash
   python test_bitnet.py --setup-only
   ```
   *This will clone the `bitnet.cpp` repository into your project folder.*

2. **Build the Framework**:
   The framework is typically built during the first model setup. However, if you encounter issues with the default compiler (Clang), you can build it manually using GCC:
   ```bash
   cd bitnet.cpp
   mkdir -p build && cd build
   CC=gcc CXX=g++ cmake ..
   make -j
   ```
   *For detailed build troubleshooting, see [BITNET_ISSUES.md](BITNET_ISSUES.md).*

3. **Install Framework Dependencies**:
   ```bash
   pip install -r bitnet.cpp/requirements.txt
   ```

---

## 4. Platform-Specific Tips

### Ubuntu/Debian
```bash
# Install build tools
sudo apt install cmake git build-essential

# Install LLVM/Clang 18
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

### Windows
- Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload.
- Ensure "C++ CMake tools for Windows" and "Git for Windows" are checked.
- Use the **Developer Command Prompt for VS 2022** for all build commands.

### macOS
```bash
brew install cmake llvm git
```

---

## 5. Verification

To verify that your environment is correctly configured:

**For Standard Training:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**For BitNet Inference:**
```bash
python test_bitnet.py --setup-only
```
If you see `✓ bitnet.cpp setup complete`, you are ready to go!
