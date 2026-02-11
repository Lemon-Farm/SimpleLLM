# SimpleLLM

A minimal GPT-style language model implemented from scratch in PyTorch.  
This project covers the full pipeline from **raw text preprocessing → tokenization → dataset construction → Transformer architecture → training and visualization**.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset & Preprocessing](#dataset--preprocessing)  
3. [Model Architecture](#model-architecture)  
4. [Training Setup](#training-setup)  
5. [Results](#results)  
6. [Reference](#reference)  

---

## 🧭 Project Overview

This notebook implements a simplified Transformer-based language model inspired by GPT architectures.  
The goal is to understand and reproduce core LLM components at a low level without relying on high-level pretrained APIs.

### Key Features

- Custom dataset generation using sliding window strategy
- Multi-head self-attention implemented from scratch
- Configurable Transformer depth and embedding size
- End-to-end training loop with loss visualization

---

## 📚 Dataset & Preprocessing

**Source Text:**  
Sherlock Holmes stories (plain text)

### Preprocessing Steps

- Normalize whitespace and line breaks
- Remove leading/trailing spaces
- Clean title/header sections
- Tokenize using `tiktoken`
- Generate overlapping training samples

### Sliding Window Strategy

- `max_length = 64`
- `stride = 4`
- Produces overlapping `(input_ids, labels)` pairs for next-token prediction

This enables dense supervision and efficient language modeling.

---

## 🏗 Model Architecture

A simplified GPT-style Transformer with the following configuration:

- Context length: 128  
- Embedding dimension: 768  
- Number of attention heads: 12  
- Number of layers: 6  
- Dropout rate: 0.1  
- Vocabulary size: determined by tokenizer  

### Core Components

- Token and positional embeddings
- Multi-head self-attention (custom implementation)
- Feed-forward network
- Residual connections with LayerNorm
- Causal masking for autoregressive training
- Final linear projection to vocabulary space

**Objective:**  
Next-token prediction using cross-entropy loss.

---

## ⚙ Training Setup

- Optimizer: AdamW  
- Learning rate: 4e-4  
- Weight decay: 0.01  
- Batch size: 96  
- Device: CUDA (if available)  

Training loss is recorded per epoch and visualized to monitor convergence.

---

## 📈 Results

- Stable loss convergence during training
- Functional autoregressive behavior after training
- Demonstrates end-to-end Transformer-based language modeling

---

## 📎 Reference

Model structure inspired by:

*Build a Large Language Model (From Scratch)*  
https://www.manning.com/books/build-a-large-language-model-from-scratch

---

## 🧠 Technical Focus

- Low-level Transformer implementation
- Custom data pipeline construction
- Autoregressive language modeling
- Training stability and optimization dynamics
