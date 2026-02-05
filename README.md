# Aurora Fine-Tuning Project

Fine-tune Phi-3-mini model on custom book data to create Aurora.

## Prerequisites

1. **Python 3.10+** installed
2. **CUDA-capable GPU** (recommended: 16GB+ VRAM)
3. **Ollama** installed (for final deployment)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place all your training text files (.txt) in the `books/` folder:
```
books/
├── book1.txt
├── book2.txt
└── book3.txt
```

## Training Process

### Step 1: Train the Model
```bash
python train.py
```

This will:
- Load Phi-3-mini base model
- Prepare your book data
- Fine-tune with LoRA
- Save the Aurora model

Training time: 2-6 hours (depending on data size and GPU)

### Step 2: Convert to GGUF (for Ollama)
```bash
python convert_to_gguf.py
```

This converts the model to GGUF format compatible with Ollama.

### Step 3: Import to Ollama
```bash
python import_to_ollama.py
```

This creates the Aurora model in Ollama.

### Step 4: Test Aurora
```bash
python test_aurora.py
```

Or use Ollama directly:
```bash
ollama run aurora
```

## Configuration

Edit `config.py` to customize:
- Training parameters (epochs, batch size, learning rate)
- LoRA configuration (r, alpha, dropout)
- Model name and system prompt
- Data paths

## Project Structure
```
aurora/
├── books/                      # Your training data (*.txt files)
├── config.py                   # Configuration settings
├── data_preparation.py         # Data loading and preprocessing
├── train.py                    # Main training script
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
├── logs/                     # Training logs
├── data_samples/             # Sample training data
└── llama.cpp/                # llama.cpp for conversion (auto-downloaded)
```

## Monitoring Training

Training logs are saved to `logs/` folder. Monitor progress:
```bash
tail -f logs/training_*.log
```

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in `config.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Reduce `MAX_SEQ_LENGTH`

### Slow Training
- Ensure GPU is being used (check with `nvidia-smi`)
- Reduce dataset size for faster iterations
- Use smaller `LORA_R` value

### Model Quality Issues
- Increase `NUM_EPOCHS`
- Add more training data
- Adjust `LEARNING_RATE`

## Notes

- The original `phi3:mini` model in Ollama remains unchanged
- Aurora is created as a separate model
- You can run both models side by side
- Training progress is logged to both console and log files

## Support

For issues or questions, check:
- Unsloth documentation: https://github.com/unslothai/unsloth
- Ollama documentation: https://ollama.ai/