# Advanced Training Guide: Interdisciplinary Technical Mastery

This guide explains how to train Aurora on specialized technical domains: Mathematics, Circuit Designing, Mechanics, and Computational Engineering (PicoGK/RobotCEM).

## 1. Automated Data Acquisition

Aurora's data preparation pipeline automatically fetches and formats high-quality technical data. The sources are configured in `config.py`.

### Domain Coverage
The system fetches data from:
- **Mathematics**: Theoretical proofs, advanced calculus, and problem-solving.
- **Circuit Designing**: Electronic schematics, circuit theory, and hardware description languages (Verilog).
- **Mechanics**: Statics, dynamics, fluid mechanics, and materials science.
- **Biology**: Practical and knowledge on biological constraints and processes.
- **Physics**: Theoretical and practical understanding.
- **Chemistry**: Theoretical proofs, and advanced analysis of chemical problems.
- **Computational Engineering**: The full LEAP71 suite and RobotCEM implementation.

## 2. High-Quality Dataset Resources

We have selected the following datasets on Hugging Face for their technical depth:

### Mathematics
- **Camel-AI Math**: 50,000 examples of complex mathematical reasoning and problem-solving. [huggingface.co/datasets/camel-ai/math](https://huggingface.co/datasets/camel-ai/math)
- **MATH-500**: High-level competition math problems with step-by-step solutions.

### Circuit Designing & Electronics
- **Electronics StackExchange**: Real-world engineering questions and verified solutions for circuit design. [huggingface.co/datasets/gbertola/electronics-stackexchange](https://huggingface.co/datasets/gbertola/electronics-stackexchange)
- **Hardware Code**: Extracted from repositories (Verilog/VHDL) to teach Aurora hardware-level logic.

### Advanced Mechanics & Materials
- **MechanicsMaterials**: MIT-curated dataset for mechanical and manufacturing engineering. [huggingface.co/datasets/lamm-mit/MechanicsMaterials](https://huggingface.co/datasets/lamm-mit/MechanicsMaterials)
- **Camel-AI Physics**: Multi-turn technical conversations on classical and quantum mechanics. [huggingface.co/datasets/camel-ai/physics](https://huggingface.co/datasets/camel-ai/physics)

### Computational Design
- **LEAP71 Repos**: (PicoGK, ShapeKernel, LatticeLibrary) The definitive source for voxel-based computational engineering.
- **RobotCEM**: The framework that integrates Aurora with PicoGK and Blender for the Design-Simulate-Fix loop. [github.com/Purple404i/RobotCEM](https://github.com/Purple404i/RobotCEM)

### Physics
- **Camel-AI Physics**: Multi-turn technical conversations on classical and quantum mechanics. [huggingface.co/datasets/camel-ai/physics](https://huggingface.co/datasets/camel-ai/physics)

### Chemsitry
- **Camel-AI Chemistry**: Multi-turn analysis on chemistry. [huggingface.co/datasets/camel-ai/chemistry](https://huggingface.co/datasets/camel-ai/chemistry)

### Biology
- **Camel-AI Biology**: Multi-turn analysis of biological processes and features. [huggingface.co/datasets/camel-ai/physics](https://huggingface.co/datasets/camel-ai/biology)

## 3. Running the Training

1.  **Installation**:
    ```bash
    pip install datasets
    ```
2.  **Start Acquisition & Training**:
    ```bash
    python train.py
    ```
    The `data_preparation.py` script will download all configured datasets and clone the technical repositories, saving everything into the `books/` folder as structured `.txt` files.

3.  **Inspect Training Material**:
    You can look into the `books/` folder after the acquisition phase to see the raw text that will be used for training.

## 4. Fine-Tuning Strategy for Engineering

To maximize Aurora's capabilities in these domains:

- **Balanced Mix**: The default configuration provides a balanced mix. You can prioritize a domain by increasing its representation in `config.py`.
- **System Prompt Integrity**: Always use the provided `SYSTEM_PROMPT`. It contains the "Reasoning Protocol" that forces the model to calculate relevant physics and math before suggesting a design.
- **Hardware Requirements**: For advanced training with large datasets and 8k context window, an NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090/4090) is recommended.

## 5. Troubleshooting

- **Dataset Load Failures**: If a Hugging Face dataset fails to load, ensure you have a stable internet connection and have accepted any required terms on the dataset's page.
- **Git Errors**: If cloning fails, verify that `git` is installed and you have access to the public LEAP71/RobotCEM repositories.
- **Context Limits**: If you experience OOM during training on long code blocks, reduce `MAX_SEQ_LENGTH` in `config.py` to 4096.
