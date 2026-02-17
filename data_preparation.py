"""
Data preparation script - loads text files and creates training dataset.
Now supports fetching datasets from Hugging Face and LEAP71 repositories.
"""

import os
import shutil
import subprocess
import multiprocessing
from pathlib import Path
from datasets import Dataset, load_dataset
import json
from typing import List, Dict, Optional
from config import BOOKS_FOLDER, TRAIN_TEST_SPLIT, SYSTEM_PROMPT, HF_DATASETS, LEAP71_REPOS

# ── Cache folder for locally-saved Parquet files ──────────────────────────────
PARQUET_CACHE_DIR = "./parquet_cache"

# ── Per-dataset field mappings ─────────────────────────────────────────────────
# Maps repo name → function that extracts a (question, answer) tuple from one example.
# If a repo is NOT listed here, the generic fallback extractor is used.
#
# Field sources (verified from HF dataset pages):
#   camel-ai/*                         : message_1 (problem), message_2 (solution)
#   sciq                               : question, correct_answer + support paragraph
#   TIGER-Lab/MathInstruct             : instruction, output
#   HuggingFaceH4/MATH-500             : problem, solution
#   lamm-mit/MechanicsMaterials        : instruction, output  (alpaca format)
#   gbertola/electronics-stackexchange : question, answers (list)
#   GaTech-EIC/MG-Verilog              : high_level_global_summary, detailed_global_summary
#   theblackcat102/evol-codealpaca-v1  : instruction, output

DATASET_EXTRACTORS = {
    # All camel-ai science datasets share the same field names
    "camel-ai/physics":   lambda ex: (ex.get("message_1", ""), ex.get("message_2", "")),
    "camel-ai/biology":   lambda ex: (ex.get("message_1", ""), ex.get("message_2", "")),
    "camel-ai/chemistry": lambda ex: (ex.get("message_1", ""), ex.get("message_2", "")),
    "camel-ai/math":      lambda ex: (ex.get("message_1", ""), ex.get("message_2", "")),

    # SciQ: question + support paragraph + correct answer
    "sciq": lambda ex: (
        ex.get("question", ""),
        (ex.get("support", "") + "\n" + ex.get("correct_answer", "")).strip()
    ),

    # MathInstruct: standard instruction-tuning format
    "TIGER-Lab/MathInstruct": lambda ex: (ex.get("instruction", ""), ex.get("output", "")),

    # MATH-500: problem + solution
    "HuggingFaceH4/MATH-500": lambda ex: (ex.get("problem", ""), ex.get("solution", "")),

    # MechanicsMaterials: alpaca-style
    "lamm-mit/MechanicsMaterials": lambda ex: (ex.get("instruction", ""), ex.get("output", "")),

    # Electronics StackExchange: answers is a list — join all answers
    "gbertola/electronics-stackexchange": lambda ex: (
        ex.get("question", ""),
        "\n\n".join(ex.get("answers", [])) if isinstance(ex.get("answers"), list)
        else str(ex.get("answers", ""))
    ),

    # MG-Verilog: high-level summary as question, detailed summary as answer
    "GaTech-EIC/MG-Verilog": lambda ex: (
        ex.get("high_level_global_summary", ""),
        ex.get("detailed_global_summary", "")
    ),

    # EvolCodeAlpaca
    "theblackcat102/evol-codealpaca-v1": lambda ex: (ex.get("instruction", ""), ex.get("output", "")),
}


def _extract_text_from_example(repo: str, example: Dict) -> str:
    """
    Extract a formatted Q&A string from a dataset example.
    Uses per-dataset extractors where available, with a generic fallback
    that tries common field names so nothing is silently lost.
    """
    extractor = DATASET_EXTRACTORS.get(repo)

    if extractor:
        question, answer = extractor(example)
        question = str(question).strip()
        answer = str(answer).strip()
        if question and answer:
            return f"Problem: {question}\nSolution: {answer}\n\n"
        elif answer:
            return answer + "\n\n"
        return ""

    # ── Generic fallback — tries common field name patterns ───────────────────
    if "message_1" in example and "message_2" in example:
        return f"Problem: {example['message_1']}\nSolution: {example['message_2']}\n\n"
    if "instruction" in example and "output" in example:
        text = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            text += f"Input: {example['input']}\n"
        return text + f"Output: {example['output']}\n\n"
    if "instruction" in example and "response" in example:
        return f"Instruction: {example['instruction']}\nResponse: {example['response']}\n\n"
    if "question" in example and "answer" in example:
        return f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
    if "problem" in example and "solution" in example:
        return f"Problem: {example['problem']}\nSolution: {example['solution']}\n\n"
    if "description" in example and "code" in example:
        return f"Description: {example['description']}\nCode:\n{example['code']}\n\n"
    if "text" in example:
        return example["text"] + "\n\n"

    # Last resort: dump all string fields so nothing is silently skipped
    fallback = ""
    for key, val in example.items():
        if isinstance(val, str) and val.strip():
            fallback += f"{key}: {val.strip()}\n"
    return (fallback + "\n") if fallback else ""


def fetch_hf_datasets(datasets_list: List[Dict], output_folder: str):
    """
    Fetch datasets from Hugging Face and save as text files in the books folder.

    Speed improvements:
      1. split='train[:1000]'  — only generate the 1000 examples you actually use
      2. num_proc=<cores>      — parallelise generation across all CPU cores
      3. Parquet cache         — after first run, loads in seconds from local disk
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(PARQUET_CACHE_DIR, exist_ok=True)

    num_proc = multiprocessing.cpu_count()

    for ds_info in datasets_list:
        repo = ds_info["repo"]
        name = ds_info["name"]
        print(f"Fetching HF dataset: {repo}...")

        safe_name = repo.replace("/", "_")
        parquet_path = os.path.join(PARQUET_CACHE_DIR, f"{safe_name}.parquet")

        try:
            # ── Load from local Parquet cache if available ────────────────────
            if os.path.exists(parquet_path):
                print(f"  Loading from local cache: {parquet_path}")
                dataset = load_dataset(
                    "parquet",
                    data_files={"train": parquet_path},
                    split="train",
                )
            else:
                dataset = load_dataset(
                    repo,
                    split="train[:1000]",
                    num_proc=num_proc if num_proc > 1 else None,
                )
                print(f"  Saving to local cache: {parquet_path}")
                dataset.to_parquet(parquet_path)

            # ── Extract text using the correct per-dataset field mapping ──────
            text_content = ""
            count = 0
            for example in dataset:
                if count >= 1000:
                    break
                text = _extract_text_from_example(repo, example)
                if text:
                    text_content += text
                    count += 1

            if text_content:
                file_path = os.path.join(output_folder, f"hf_{name}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                print(f"  Saved {count} examples to {file_path}")
            else:
                print(f"  WARNING: No text extracted from {repo} — check field names!")

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
        repo_name = repo_url.split("/")[-1]
        print(f"Fetching LEAP71 repo: {repo_name}...")
        repo_path = os.path.join(temp_dir, repo_name)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_path],
                check=True,
                capture_output=True,
            )

            extracted_text = f"--- REPOSITORY: {repo_name} ---\n"
            file_count = 0
            valid_extensions = (".md", ".cs", ".py", ".v", ".sv", ".cpp", ".h", ".txt")

            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(valid_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read().strip()
                                if content:
                                    extracted_text += f"\n\n--- Source File: {file} ---\n\n"
                                    extracted_text += content
                                    file_count += 1
                        except Exception:
                            continue

            if file_count > 0:
                output_file = os.path.join(output_folder, f"leap71_{repo_name}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                print(f"  Extracted {file_count} files to {output_file}")

        except Exception as e:
            print(f"  Error processing {repo_url}: {str(e)}")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def read_books(folder_path: str) -> List[Dict]:
    """Read all text files from the books folder."""
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
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    books.append({"filename": file_path.name, "content": content})
        except Exception as e:
            print(f"  Error reading {file_path.name}: {str(e)}")

    return books


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Split text into overlapping chunks."""
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
    """Create training examples from books in chat format."""
    training_examples = []

    for book in books:
        content = book["content"]
        filename = book["filename"]

        if filename.startswith("hf_"):
            user_prompt = f"Explain the technical concepts related to {filename[3:-4]}."
        elif filename.startswith("leap71_"):
            user_prompt = f"Provide examples and documentation for {filename[7:-4]} from LEAP71."
        else:
            user_prompt = f"Tell me about the content from {filename}."

        chunks = chunk_text(content)
        for chunk in chunks:
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": chunk},
                ]
            }
            training_examples.append(example)

    return training_examples


def format_for_training(example: Dict, tokenizer=None) -> Dict:
    """Format examples for training with proper chat template."""
    messages = example["messages"]

    if tokenizer:
        try:
            return {
                "text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            }
        except Exception:
            pass

    formatted_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted_text += f"<|{role}|>\n{content}<|end|>\n"

    return {"text": formatted_text}


def prepare_dataset(tokenizer=None, fetch_remote=True):
    """Main function to prepare the complete dataset."""
    print("=" * 60)
    print("AURORA DATA PREPARATION")
    print("=" * 60)

    if fetch_remote:
        print("\n0. Fetching remote data...")
        fetch_hf_datasets(HF_DATASETS, BOOKS_FOLDER)
        fetch_leap71_data(LEAP71_REPOS, BOOKS_FOLDER)

    print(f"\n1. Reading training data from '{BOOKS_FOLDER}' folder...")
    books = read_books(BOOKS_FOLDER)

    if not books:
        print("\nWARNING: No training data found.")
        books = [{"filename": "dummy.txt", "content": "Aurora is a technical AI assistant."}]

    print(f"\nSuccessfully loaded {len(books)} source(s)")

    print("\n2. Creating training examples...")
    training_examples = create_training_examples(books)
    print(f"Created {len(training_examples)} training examples")

    print("\n3. Formatting examples for training...")
    formatted_examples = [format_for_training(ex, tokenizer) for ex in training_examples]

    dataset = Dataset.from_list(formatted_examples)

    print(
        f"\n4. Splitting dataset (train: {TRAIN_TEST_SPLIT*100}%, val: {(1-TRAIN_TEST_SPLIT)*100}%)..."
    )
    split_dataset = dataset.train_test_split(test_size=1 - TRAIN_TEST_SPLIT, seed=42)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    os.makedirs("./data_samples", exist_ok=True)
    with open("./data_samples/sample_training_example.json", "w", encoding="utf-8") as f:
        json.dump(training_examples[0], f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = prepare_dataset(fetch_remote=True)
    print("\nDataset ready for training!")