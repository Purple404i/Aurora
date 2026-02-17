# Advanced Training Guide: Interdisciplinary Technical Mastery

This guide provides deep technical details on how to train Aurora to become an expert in Robotics, Physics, Math, and Computational Engineering.

---

## 1. Technical Data Pipeline
**Primary Script**: `data_preparation.py`

Aurora's intelligence comes from high-quality data. The acquisition process is automated to ensure consistency.

### Data Sources
1.  **Hugging Face Hub**: Fetches structured Q&A and technical datasets.
    - *Example*: `camel-ai/math` for complex reasoning.
    - *Example*: `GaTech-EIC/MG-Verilog` for circuit design and hardware logic.
2.  **Technical Repositories**: Clones professional engineering frameworks.
    - **PicoGK**: Voxel-based geometry engine.
    - **LEAP71 ShapeKernel**: Foundation for computational design.
    - **RobotCEM**: The framework that uses Aurora for physics-driven robotics.

### How to Execute Acquisition
Run the following command to populate your local knowledge base:
```bash
python data_preparation.py
```
This will create a structured `books/` folder:
- `hf_math.txt`: Combined mathematical proofs.
- `leap71_PicoGK.txt`: PicoGK C# syntax and documentation.
- `robotcem_RobotCEM.txt`: RobotCEM workflow logic.

---

## 2. Specialized Fine-Tuning Strategy

### Mathematics & Symbology
Aurora uses `MATH-500` and `MathInstruct` to learn the **Reasoning Protocol**. Instead of just guessing, the model learns to:
- Formulate an equation.
- Solve it step-by-step using `<thinking>` tags.
- Provide a validated technical solution.

### Circuit Design & EDA
By training on `MG-Verilog` and `Electronics StackExchange`, Aurora gains the ability to:
- Generate synthesizable Verilog modules.
- Recommend specific components (ESP32, STM32, MOSFETs) for robotic control.
- Design power-efficient battery management systems.

### Computational Geometry (LEAP71 & PicoGK)
Aurora is trained specifically on the **LEAP71 repository codebase**. This enables it to write valid C# code for:
- Creating complex 3D lattices.
- Defining functional shapes (ShapeKernels).
- Integrating designs directly with voxel-based simulation.

---

## 3. Optimizing the Workflow

To get the best results for a "RobotCEM" type assistant:

1.  **Context Window**: Always set `MAX_SEQ_LENGTH` to at least `4096` in `train.py`. High-end hardware should use `8192`.
2.  **Dataset Balance**: Ensure your `HF_DATASETS` list in `config.py` includes a mix of Math, Science, and Coding.
3.  **The "Aurora" Transformation**:
    - **Merging**: Use `convert_to_gguf.py` to bake your new knowledge into a single model file.
    - **Deployment**: Use `import_to_ollama.py` to give the model its "Aurora" identity via the custom system prompt.

---

## 4. Hardware Recommendations

| Phase | Recommendation |
| :--- | :--- |
| **Data Fetching** | Any machine with internet. |
| **Standard Fine-Tuning** | NVIDIA GPU (24GB VRAM like 3090/4090). |
| **BitNet Training** | NVIDIA GPU (8GB+ VRAM). |
| **Inference (Aurora)** | Standard CPU (8+ cores) using bitnet.cpp. |

---

## 5. Troubleshooting the Pipeline

- **Failed Downloads**: If `data_preparation.py` fails to fetch a dataset, check your internet connection and ensure `datasets` library is up to date (`pip install -U datasets`).
- **Incomplete Extraction**: If code examples are missing, ensure the file extensions are listed in the `valid_extensions` list within `data_preparation.py`.
- **Merging Errors**: If `convert_to_gguf.py` fails, ensure you have sufficient disk space (typically 2x the model size).
