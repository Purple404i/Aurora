"""
Main training script for fine-tuning ollama hugging face models using unsloth to create Aurora
"""

import os
import torch
import subprocess
import importlib
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from data_preparation import prepare_dataset
from config import *
import logging
from datetime import datetime

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
log_filename = os.path.join(LOGS_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_ollama_models():
    """
    Get the list of locally downloaded Ollama models.
    """
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1: # Header only or empty
            return []
            
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []

def map_ollama_to_unsloth(ollama_model):
    """
    Map Ollama model names to Unsloth-compatible Hugging Face model names.
    """
    ollama_model_lower = ollama_model.lower()
    
    # Common mappings
    mapping = {
        "phi3": "unsloth/Phi-3-mini-4k-instruct",
        "phi3:mini": "unsloth/Phi-3-mini-4k-instruct",
        "phi3:medium": "unsloth/Phi-3-medium-4k-instruct",
        "llama3": "unsloth/llama-3-8b-bnb-4bit",
        "llama3:8b": "unsloth/llama-3-8b-bnb-4bit",
        "llama3:70b": "unsloth/llama-3-70b-bnb-4bit",
        "mistral": "unsloth/mistral-7b-v0.3-bnb-4bit",
        "gemma": "unsloth/gemma-7b-bnb-4bit",
        "gemma:7b": "unsloth/gemma-7b-bnb-4bit",
        "gemma:2b": "unsloth/gemma-2b-bnb-4bit",
        "llama2": "unsloth/llama-2-7b-bnb-4bit",
    } 
    # More models at https://huggingface.co/unsloth

    
    # Check for direct match
    if ollama_model_lower in mapping:
        return mapping[ollama_model_lower]
    
    # Check for base name match (e.g., llama3:latest -> llama3)
    base_name = ollama_model_lower.split(':')[0]
    if base_name in mapping:
        return mapping[base_name]
    else:
        print(f"\nWarning: No mapping found for '{ollama_model}'. Please make sure its a Hugging Face model name and provided by unsloth.")
        print(f"You can get available models' name https://huggingface.co/unsloth")
        print("If model is present in both the places add the required model's name from https://huggingface.co/unsloth and then add it to the mappings in the train.py")
    return None

def update_config_file(new_model_name):
    """
    Update the BASE_MODEL_NAME in config.py.
    """
    config_path = 'config.py'
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    updated = False
    with open(config_path, 'w') as f:
        for line in lines:
            if line.strip().startswith('BASE_MODEL_NAME ='):
                f.write(f'BASE_MODEL_NAME = "{new_model_name}"\n')
                updated = True
            else:
                f.write(line)
    return updated

def load_model():
    """
    Load the base model with LoRA configuration.
    
    Returns:
        model: The prepared model for fine-tuning
        tokenizer: The tokenizer
    """
    logger.info("="*60)
    logger.info("LOADING BASE MODEL")
    logger.info("="*60)
    logger.info(f"Base model: {BASE_MODEL_NAME}")
    logger.info(f"Max sequence length: {MAX_SEQ_LENGTH}")
    logger.info(f"Using 4-bit quantization: {USE_4BIT}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=USE_4BIT,
    )
    
    logger.info("Base model loaded successfully!")
    
    logger.info("\n" + "="*60)
    logger.info("APPLYING LoRA CONFIGURATION")
    logger.info("="*60)
    logger.info(f"LoRA r: {LORA_R}")
    logger.info(f"LoRA alpha: {LORA_ALPHA}")
    logger.info(f"LoRA dropout: {LORA_DROPOUT}")
    logger.info(f"Target modules: {TARGET_MODULES}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=42,
    )
    
    logger.info("LoRA configuration applied successfully!")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nTrainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

def train_aurora():
    """
    Main training function.
    """
    logger.info("\n" + "="*60)
    logger.info("AURORA FINE-TUNING STARTED")
    logger.info("="*60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model()
    
    # Prepare dataset
    logger.info("\n" + "="*60)
    logger.info("PREPARING DATASET")
    logger.info("="*60)
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
    # Training arguments
    logger.info("\n" + "="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Number of epochs: {NUM_EPOCHS}")
    logger.info(f"Warmup steps: {WARMUP_STEPS}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        max_grad_norm=MAX_GRAD_NORM,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    trainer.train()
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    # Save final model
    logger.info("\n" + "="*60)
    logger.info("SAVING AURORA MODEL")
    logger.info("="*60)
    logger.info(f"Saving to: {OUTPUT_DIR}")
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Model saved successfully!")
    
    # Save merged model (LoRA weights merged into base model)
    logger.info("\nSaving merged model...")
    merged_output_dir = f"{OUTPUT_DIR}_merged"
    
    model.save_pretrained_merged(
        merged_output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    
    logger.info(f"Merged model saved to: {merged_output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("AURORA FINE-TUNING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nYour Aurora model is ready!")
    logger.info(f"Location: {OUTPUT_DIR}")
    logger.info(f"Merged model: {merged_output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Convert to GGUF format (see convert_to_gguf.py)")
    logger.info(f"2. Import to Ollama (see import_to_ollama.py)")

if __name__ == "__main__":
    try:
        # 1. Check for Ollama models
        ollama_models = get_ollama_models()
        
        if ollama_models:
            print("\n" + "="*60)
            print("LOCAL OLLAMA MODELS FOUND")
            print("="*60)
            for i, model in enumerate(ollama_models, 1):
                print(f"{i}. {model}")
            print(f"{len(ollama_models) + 1}. Keep current ({BASE_MODEL_NAME})")
            
            choice = input(f"\nSelect a model to finetune (1-{len(ollama_models) + 1}): ")
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(ollama_models):
                    selected_ollama_model = ollama_models[choice_idx]
                    logger.info(f"Selected Ollama model: {selected_ollama_model}")
                    
                    # 2. Map to Unsloth/HF model
                    unsloth_model = map_ollama_to_unsloth(selected_ollama_model)
                    
                    if not unsloth_model:
                        print(f"\nWarning: No direct Unsloth mapping found for '{selected_ollama_model}'")
                        unsloth_model = input(f"Please enter the Hugging Face model name to use (or press Enter to use '{selected_ollama_model}' as is): ").strip()
                        if not unsloth_model:
                            unsloth_model = selected_ollama_model
                    
                    # 3. Update config.py
                    if update_config_file(unsloth_model):
                        logger.info(f"Updated config.py with BASE_MODEL_NAME = \"{unsloth_model}\"")
                        
                        # 4. Reload configuration
                        import config
                        importlib.reload(config)
                        import data_preparation
                        importlib.reload(data_preparation)
                        
                        # Update globals in this module
                        globals().update({k: v for k, v in vars(config).items() if not k.startswith('_')})
                        
                        # Specifically re-import prepare_dataset as it might be using old config values
                        from data_preparation import prepare_dataset
                        
                        logger.info("Configuration reloaded successfully")
                else:
                    logger.info("Keeping current model configuration")
            except ValueError:
                logger.info("Invalid choice, keeping current model configuration")
        else:
            logger.info("No Ollama models found or Ollama not running, using default configuration")

        # Start training
        train_aurora()
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise