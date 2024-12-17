## Introduction

Inspired by the success of **Chinese LLaMA**, we aim to explore whether **expanding the tokenizer vocabulary** can improve model performance in vertical domains. By utilizing **custom corpora** and training new vocabularies with **Unigram** and **BPE algorithms**, we generate expanded tokenizers and merge them into pre-existing models. 

The goal is to test if a specialized vocabulary trained on domain-specific data can better capture domain-specific patterns and semantics, ultimately enhancing downstream task performance.

---

## Directory Structure

The directory is organized as follows:

```plaintext
tokenizer/
│
├── merge_tokenizer.py       # Merge a custom SentencePiece tokenizer into an existing model.
├── readme.md                # Documentation for the project.
├── test.py                  # Test script for evaluating tokenizer performance.
├── token_eff_test.py        # Script for testing tokenization efficiency.
├── train_tokenizer.py       # Train a new tokenizer on domain-specific data.
└── vacab.txt                # Extracted vocabulary list from the trained tokenizer.
```

---

## File Descriptions

### 1. **merge_tokenizer.py**
   - **Purpose**: Merges the custom-trained tokenizer vocabulary into an existing pre-trained tokenizer.
   - **Functionality**:
     - Loads a pre-existing tokenizer and a newly trained SentencePiece tokenizer.
     - Adds the new tokens to the original tokenizer.
     - Saves the updated tokenizer for further use.

---

### 2. **train_tokenizer.py**
   - **Purpose**: Trains a tokenizer on domain-specific corpora using SentencePiece.
   - **Functionality**:
     - Supports **Unigram** and **BPE** algorithms for vocabulary generation.
     - Configurable vocabulary size and character coverage.
     - Outputs trained tokenizer models (`.model` and `.vocab` files).

---

### 3. **token_eff_test.py**
   - **Purpose**: Tests the efficiency of the tokenizer.
   - **Functionality**:
     - Measures the encoding speed and tokenization performance on randomly generated text or custom input data.
     - Reports metrics like **compression ratio**, **tokenization speed**, and total token count.

---

### 4. **test.py**
   - **Purpose**: Tests and evaluates the quality of the tokenized outputs.
   - **Functionality**:
     - Loads and tokenizes specific input text using the expanded tokenizer.
     - Compares tokenized results against baseline models.

---

### 5. **vacab.txt**
   - **Purpose**: Stores the extracted vocabulary list from a trained tokenizer.
   - **Functionality**:
     - Provides a human-readable version of the tokenizer's vocabulary for analysis or debugging.

---

## Challenges

1. **Unknown Performance Gains**:
   - It is uncertain whether an expanded domain-specific vocabulary will improve model performance. Experiments and benchmarks are needed to validate the benefits.

2. **Embedding Size Adjustment**:
   - Expanding the tokenizer vocabulary requires changes to the **embedding layer size** in the model.
   - A method to initialize embeddings for new tokens must be established to maintain model stability.

3. **Vocabulary Quality**:
   - The quality of the expanded vocabulary heavily influences performance. It depends on:
     - **The data quality** of the training corpus.
     - **The efficiency of the tokenization algorithm** (Unigram vs. BPE).

4. **Balancing Token Coverage**:
   - Overfitting to domain-specific tokens may reduce generalization capability.
   - Proper token coverage and trade-offs must be carefully considered.

