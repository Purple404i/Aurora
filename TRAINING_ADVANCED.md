# Advanced Training Guide: Interdisciplinary Technical Mastery

This guide explains how to train Aurora on specialized technical domains: Mathematics, Circuit Designing, Mechanics, and Computational Engineering (PicoGK/RobotCEM).

## 1. Automated Data Acquisition

Aurora's data preparation pipeline automatically fetches and formats high-quality technical data. The sources are configured in `config.py`.

### Domain Coverage
- **Mathematics**: Advanced reasoning, proofs, and symbolic manipulation.
- **Circuit Designing**: EDA (Electronic Design Automation), PCB layout theory, and Hardware Description Languages (Verilog/SystemVerilog).
- **Advanced Mechanics**: Finite Element Analysis (FEA) theory, Computational Fluid Dynamics (CFD), and Kinematic modeling.
- **Computational Engineering**: The full LEAP71 suite (Voxel-based geometry) and RobotCEM integration.

## 2. Technical Domain Deep-Dive

### Mathematics & Symbolic Reasoning
Aurora is trained on datasets like **Camel-AI Math** and **MATH-500** to move beyond simple arithmetic. It learns to solve:
- **Differential Equations**: For modeling physical systems.
- **Linear Algebra**: For kinematic transformations and rotation matrices.
- **Optimization**: For finding optimal robot paths and lattice structures.

### Circuit Designing & Hardware (Verilog)
By integrating the **MG-Verilog** and **Electronics StackExchange** datasets, Aurora can:
- **Generate RTL Code**: Write synthesizable Verilog for custom hardware accelerators.
- **Design Schematics**: Suggest components and connection logic for robot control boards (ESP32, STM32).
- **Debug Hardware**: Analyze circuit failure modes and suggest fixes based on engineering best practices.

### Advanced Mechanics & Materials Science
Using the **MIT MechanicsMaterials** dataset, Aurora gains insights into:
- **Stress-Strain Analysis**: Predicting where a 3D-printed part might fail.
- **Material Selection**: Choosing between PLA, ABS, Carbon Fiber, or Biopolymers based on environmental and mechanical constraints.
- **Kinematics**: Designing 6-DOF arms with accurate Jacobian calculations.

## 3. High-Quality Dataset Resources

We have selected the following datasets for their technical depth:

### Mathematics & Science
- **Camel-AI Math**: 50,000 examples of complex mathematical reasoning. [huggingface.co/datasets/camel-ai/math](https://huggingface.co/datasets/camel-ai/math)
- **MATH-500**: High-level competition math problems with solutions. [huggingface.co/datasets/HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- **SciQ**: 13,000+ science exam questions. [huggingface.co/datasets/sciq](https://huggingface.co/datasets/sciq)
- **Camel-AI Physics/Biology/Chemistry**: Multi-turn technical conversations.

### Circuits & Mechanics
- **MG-Verilog**: Multi-grained dataset for hardware design. [huggingface.co/datasets/GaTech-EIC/MG-Verilog](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog)
- **Electronics StackExchange**: Real-world engineering Q&A. [huggingface.co/datasets/gbertola/electronics-stackexchange](https://huggingface.co/datasets/gbertola/electronics-stackexchange)
- **MIT MechanicsMaterials**: Mechanical engineering dataset. [huggingface.co/datasets/lamm-mit/MechanicsMaterials](https://huggingface.co/datasets/lamm-mit/MechanicsMaterials)

### Computational Design
- **LEAP71 Repos**: The definitive source for PicoGK voxel-based engineering.
- **RobotCEM**: Aurora's primary integration framework. [github.com/Purple404i/RobotCEM](https://github.com/Purple404i/RobotCEM)

## 4. Running the Training

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

## 5. Fine-Tuning Strategy for Engineering

To maximize Aurora's capabilities in these domains:

- **Balanced Mix**: The default configuration provides a balanced mix. You can prioritize a domain by increasing its representation in `config.py`.
- **System Prompt Integrity**: Always use the provided `SYSTEM_PROMPT`. It contains the "Reasoning Protocol" that forces the model to calculate relevant physics and math before suggesting a design.
- **Hardware Requirements**: For advanced training with large datasets and 8k context window, an NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090/4090) is recommended.

## 6. Troubleshooting

- **Dataset Load Failures**: If a Hugging Face dataset fails to load, ensure you have a stable internet connection and have accepted any required terms on the dataset's page.
- **Git Errors**: If cloning fails, verify that `git` is installed and you have access to the public LEAP71/RobotCEM repositories.
- **Context Limits**: If you experience OOM during training on long code blocks, reduce `MAX_SEQ_LENGTH` in `config.py` to 4096.
