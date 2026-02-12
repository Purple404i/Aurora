"""
Data preparation script - loads text files and creates training dataset.
Now supports fetching datasets from Hugging Face and LEAP71 repositories.
"""

import os
import shutil
import subprocess
from pathlib import Path
from datasets import Dataset, load_dataset
import json
from typing import List, Dict, Optional
from config import BOOKS_FOLDER, TRAIN_TEST_SPLIT, SYSTEM_PROMPT, HF_DATASETS, LEAP71_REPOS

def fetch_hf_datasets(datasets_list: List[Dict], output_folder: str):
    """
    Fetch datasets from Hugging Face and save as text files in the books folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for ds_info in datasets_list:
        repo = ds_info['repo']
        name = ds_info['name']
        print(f"Fetching HF dataset: {repo}...")
        try:
            # Load the dataset
            dataset = load_dataset(repo, split='train', trust_remote_code=True)

            # Convert to text format
            text_content = ""
            count = 0
            for example in dataset:
                if count >= 1000: # Limit to 1000 examples per dataset to avoid massive files
                    break

                # Extract relevant text based on common HF dataset structures
                if 'text' in example:
                    text_content += example['text'] + "\n\n"
                elif 'instruction' in example and 'output' in example:
                    text_content += f"Instruction: {example['instruction']}\n"
                    if 'input' in example and example['input']:
                        text_content += f"Input: {example['input']}\n"
                    text_content += f"Output: {example['output']}\n\n"
                elif 'question' in example and 'answer' in example:
                    text_content += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
                elif 'instruction' in example and 'response' in example:
                    text_content += f"Instruction: {example['instruction']}\nResponse: {example['response']}\n\n"

                count += 1

            if text_content:
                file_path = os.path.join(output_folder, f"hf_{name}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"  Successfully saved {count} examples to {file_path}")
        except Exception as e:
            print(f"  Error fetching {repo}: {str(e)}")

def fetch_leap71_data(repos_list: List[str], output_folder: str):
    """
    Clone LEAP71 repositories and extract documentation and code examples.
    """
    os.makedirs(output_folder, exist_ok=True)
    temp_dir = "temp_repos"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    for repo_url in repos_list:
        repo_name = repo_url.split('/')[-1]
        print(f"Fetching LEAP71 repo: {repo_name}...")
        repo_path = os.path.join(temp_dir, repo_name)
        
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_path],
                           check=True, capture_output=True)

            # Extract .md and .cs (C# for PicoGK) and .py files
            extracted_text = f"--- REPOSITORY: {repo_name} ---\n"
            file_count = 0
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(('.md', '.cs', '.py')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    extracted_text += f"\n\n--- Source File: {file} ---\n\n"
                                    extracted_text += content
                                    file_count += 1
                        except:
                            continue

            if file_count > 0:
                output_file = os.path.join(output_folder, f"leap71_{repo_name}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"  Extracted {file_count} files to {output_file}")

        except Exception as e:
            print(f"  Error processing {repo_url}: {str(e)}")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def read_books(folder_path: str) -> List[Dict]:
    """
    Read all text files from the books folder.
    """
    books = []
    folder = Path(folder_path)
    
    if not folder.exists():
        os.makedirs(folder_path, exist_ok=True)
        return []
    
    txt_files = list(folder.glob("*.txt"))
    
    if not txt_files:
        return []
    
    print(f"Found {len(txt_files)} text file(s) in {folder_path}")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    books.append({
                        'filename': file_path.name,
                        'content': content
                    })
        except Exception as e:
            print(f"  Error reading {file_path.name}: {str(e)}")
    
    return books

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """
    Split text into overlapping chunks.
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
    Create training examples from books in chat format.
    """
    training_examples = []
    
    for book in books:
        content = book['content']
        filename = book['filename']
        
        # Determine appropriate prompt based on source
        if filename.startswith('hf_'):
            user_prompt = f"Explain the technical concepts related to {filename[3:-4]}."
        elif filename.startswith('leap71_'):
            user_prompt = f"Provide examples and documentation for {filename[7:-4]} from LEAP71."
        else:
            user_prompt = f"Tell me about the content from {filename}."

        # Chunk the content
        chunks = chunk_text(content)
        
        for chunk in chunks:
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": chunk}
                ]
            }
            training_examples.append(example)
    
    return training_examples

def format_for_training(example: Dict, tokenizer=None) -> Dict:
    """
    Format examples for training with proper chat template.
    """
    messages = example['messages']
    
    if tokenizer:
        try:
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
        except Exception:
            pass

    # Fallback formatting
    formatted_text = ""
    for message in messages:
        role = message['role']
        content = message['content']
        formatted_text += f"<|{role}|>\n{content}<|end|>\n"
    
    return {"text": formatted_text}

def prepare_dataset(tokenizer=None, fetch_remote=True):
    """
    Main function to prepare the complete dataset.
    """
    print("="*60)
    print("AURORA DATA PREPARATION")
    print("="*60)
    
    if fetch_remote:
        print("\n0. Fetching remote data...")
        fetch_hf_datasets(HF_DATASETS, BOOKS_FOLDER)
        fetch_leap71_data(LEAP71_REPOS, BOOKS_FOLDER)

    # Read all files in books folder
    print(f"\n1. Reading training data from '{BOOKS_FOLDER}' folder...")
    books = read_books(BOOKS_FOLDER)
    
    if not books:
        print("\nWARNING: No training data found. Please add .txt files to the books folder or enable remote fetching.")
        # Create a tiny dummy example so it doesn't crash if someone runs it empty
        dummy_books = [{'filename': 'dummy.txt', 'content': 'Aurora is a technical AI assistant.'}]
        books = dummy_books
    
    print(f"\nSuccessfully loaded {len(books)} source(s)")
    
    # Create training examples
    print("\n2. Creating training examples...")
    training_examples = create_training_examples(books)
    print(f"Created {len(training_examples)} training examples")
    
    # Format for training
    print("\n3. Formatting examples for training...")
    formatted_examples = [format_for_training(ex, tokenizer) for ex in training_examples]
    
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
    os.makedirs("./data_samples", exist_ok=True)
    with open("./data_samples/sample_training_example.json", 'w', encoding='utf-8') as f:
        json.dump(training_examples[0], f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = prepare_dataset(fetch_remote=True)
    print("\nDataset ready for training!")
