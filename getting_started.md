# Getting Started with Aurora Fine-Tuning

This guide will walk you through the process of fine-tuning a custom AI model using your own data and local Ollama models.

## 1. Preparation

### Install Dependencies
Make sure you have all required Python packages installed:
```bash
pip install -r requirements.txt
```

### Organize Your Data
Place your training text files in the `books/` directory. The script will automatically process all `.txt` files in this folder.
```
books/
├── mechanics_guide.txt
├── robotics_advanced.txt
└── biology_systems.txt
```

## 2. Starting the Training

Run the main training script:
```bash
python train.py
```

### Interactive Setup

When you start the script, it will perform the following steps:

1.  **Model Selection**: It detects all models you have already downloaded via Ollama. You will see a list and can choose which one you want to fine-tune (e.g., Llama-3, Mistral, or Phi-3).
2.  **Sequence Length**: You will be prompted to enter the `MAX_SEQ_LENGTH`. This determines the context window for training. Common values are 2048 or 4096.
3.  **Automatic Detection**: The script automatically inspects the chosen model to find the correct internal layer names (`TARGET_MODULES`) for LoRA. This ensures the fine-tuning "unsloth mechanism" works perfectly with different architectures.
4.  **Configuration Update**: Your choices are saved to `config.py` for persistence, but your custom `SYSTEM_PROMPT` is always preserved.

## 3. How it Works Under the Hood

### Dynamic Chat Templates
Aurora supports multiple model architectures. To ensure the model learns correctly, `data_preparation.py` uses the selected model's native chat template. This means if you pick Llama-3, it uses Llama-3's format; if you pick Phi-3, it uses Phi-3's tags.

### Efficient Fine-Tuning
We use the **Unsloth** library and **LoRA** (Low-Rank Adaptation) to make fine-tuning fast and memory-efficient. This allows you to train powerful models on consumer-grade GPUs.

## 4. Post-Training

Once training is complete, your model will be saved in `aurora_output/`. You can then:

1.  **Convert to GGUF**: Run `python convert_to_gguf.py`.
2.  **Import to Ollama**: Run `python import_to_ollama.py`.
3.  **Test**: Use `python test_aurora.py` or run `ollama run aurora`.

## Tips for Success

- **VRAM Management**: If you run out of memory (OOM), try reducing the `MAX_SEQ_LENGTH` or the `BATCH_SIZE` in `config.py`.
- **Model Choice**: Llama-3 and Mistral are excellent for general reasoning, while Phi-3 is very efficient for its size.
- **Data Quality**: The quality of your fine-tuned model depends heavily on the quality and quantity of text files in your `books/` folder.
