"""
Main training script for fine-tuning Phi-3-mini to create Aurora
"""

import os
import torch
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

def load_model():
    """
    Load the base Phi-3-mini model with LoRA configuration.
    
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
    train_dataset, val_dataset = prepare_dataset()
    
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
        train_aurora()
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise