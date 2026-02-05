"""
Data preparation script - loads text files and creates training dataset
"""

import os
from pathlib import Path
from datasets import Dataset
import json
from typing import List, Dict
from config import BOOKS_FOLDER, TRAIN_TEST_SPLIT, SYSTEM_PROMPT

def read_books(folder_path: str) -> List[str]:
    """
    Read all text files from the books folder.
    
    Args:
        folder_path: Path to the folder containing book text files
        
    Returns:
        List of text contents from all books
    """
    books = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Books folder not found: {folder_path}")
    
    txt_files = list(folder.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder_path}")
    
    print(f"Found {len(txt_files)} text file(s) in {folder_path}")
    
    for file_path in txt_files:
        print(f"Reading: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                books.append({
                    'filename': file_path.name,
                    'content': content
                })
    
    return books

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += chunk_size - overlap
    
    return chunks

def create_training_examples(books: List[Dict]) -> List[Dict]:
    """
    Create training examples from books.
    
    Args:
        books: List of book dictionaries with content
        
    Returns:
        List of training examples in chat format
    """
    training_examples = []
    
    for book in books:
        content = book['content']
        filename = book['filename']
        
        # Chunk the book content
        chunks = chunk_text(content, chunk_size=1500, overlap=300)
        
        print(f"  Created {len(chunks)} chunks from {filename}")
        
        for i, chunk in enumerate(chunks):
            # Create a simple Q&A format for training
            # This format helps the model learn the content
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Tell me about the content from {filename}."},
                    {"role": "assistant", "content": chunk}
                ]
            }
            training_examples.append(example)
    
    return training_examples

def format_for_training(example: Dict) -> Dict:
    """
    Format examples for training with proper chat template.
    
    Args:
        example: Training example with messages
        
    Returns:
        Formatted example with text field
    """
    messages = example['messages']
    
    # Format as chat
    formatted_text = ""
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == "system":
            formatted_text += f"<|system|>\n{content}<|end|>\n"
        elif role == "user":
            formatted_text += f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            formatted_text += f"<|assistant|>\n{content}<|end|>\n"
    
    return {"text": formatted_text}

def prepare_dataset():
    """
    Main function to prepare the complete dataset.
    
    Returns:
        Train and validation datasets
    """
    print("="*60)
    print("AURORA DATA PREPARATION")
    print("="*60)
    
    # Read all books
    print(f"\n1. Reading books from '{BOOKS_FOLDER}' folder...")
    books = read_books(BOOKS_FOLDER)
    
    if not books:
        raise ValueError("No books were loaded. Please add .txt files to the books folder.")
    
    print(f"\nSuccessfully loaded {len(books)} book(s)")
    total_chars = sum(len(book['content']) for book in books)
    print(f"Total characters: {total_chars:,}")
    
    # Create training examples
    print("\n2. Creating training examples...")
    training_examples = create_training_examples(books)
    print(f"Created {len(training_examples)} training examples")
    
    # Format for training
    print("\n3. Formatting examples for training...")
    formatted_examples = [format_for_training(ex) for ex in training_examples]
    
    # Create dataset
    dataset = Dataset.from_list(formatted_examples)
    
    # Split into train and validation
    print(f"\n4. Splitting dataset (train: {TRAIN_TEST_SPLIT*100}%, val: {(1-TRAIN_TEST_SPLIT)*100}%)...")
    split_dataset = dataset.train_test_split(test_size=1-TRAIN_TEST_SPLIT, seed=42)
    
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Save sample
    print("\n5. Saving sample data...")
    os.makedirs("./data_samples", exist_ok=True)
    with open("./data_samples/sample_training_example.json", 'w', encoding='utf-8') as f:
        json.dump(training_examples[0], f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = prepare_dataset()
    print("\nDataset ready for training!")
    print(f"\nSample text (first 500 chars):")
    print("-"*60)
    print(train_dataset[0]['text'][:500])
    print("-"*60)