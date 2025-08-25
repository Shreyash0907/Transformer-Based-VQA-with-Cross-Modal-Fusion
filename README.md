# Transformer-Based VQA with Cross-Modal Fusion

A multi-modal transformer architecture for Visual Question Answering on the CLEVR dataset, implementing cross-attention fusion between visual and textual features.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Features](#features)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Variants](#model-variants)
- [Evaluation](#evaluation)

## 🎯 Overview

This project implements a transformer-based Visual Question Answering system designed to answer natural language questions about images. The model combines visual features from a ResNet101 encoder with textual features from a custom transformer encoder using cross-attention mechanisms.

**Key Highlights:**
- 🏆 Best accuracy: **77.04%** on CLEVR testA dataset
- 🔄 Cross-modal attention fusion between image and text features
- 📈 Progressive improvements through fine-tuning and advanced techniques
- 🎯 Zero-shot evaluation on out-of-distribution data

## 🏗️ Architecture

The VQA model consists of four main components:

### 1. Image Encoder
- **Base Model**: Pre-trained ResNet101 from torchvision
- **Output**: Visual features of shape [B, 2048, h, w]
- **Projection**: Linear layer to embedding dimension (768)

### 2. Text Encoder
- **Architecture**: 6-layer Transformer encoder with 8 attention heads
- **Features**: 
  - Learnable [CLS] token (similar to BERT)
  - Positional embeddings
  - Token embedding dimension: 768

### 3. Feature Fusion
- **Mechanism**: Cross-attention with 8 heads
- **Process**: Text features as queries, image features as keys/values
- **Output**: Enhanced [CLS] token representation

### 4. Classifier
- **Architecture**: Two-layer MLP (768 → 500 → num_classes)
- **Activation**: ReLU
- **Output**: Answer prediction

## 📊 Dataset

**CLEVR (Compositional Language and Elementary Visual Reasoning)**
- Synthetic images with objects varying in size, shape, color, and material
- Question types: attribute identification, counting, comparison, spatial relationships
- Two variants used:
  - **CLEVR-A**: Training and primary evaluation
  - **CLEVR-B**: Zero-shot transfer evaluation

## ✨ Features

- **Multi-modal Fusion**: Cross-attention mechanism for visual-textual integration
- **Progressive Training**: From frozen to fine-tuned image encoder
- **Advanced Techniques**:
  - Focal Loss for handling class imbalance
  - BERT embedding initialization
- **Comprehensive Evaluation**: Including zero-shot transfer learning
- **Visualization Tools**: Error analysis and prediction visualization

## 📈 Results

| Model Variant | Test Accuracy | Key Features |
|---------------|---------------|--------------|
| Base Model (Frozen ResNet) | 72.57% | Frozen image encoder |
| Fine-tuned Image Encoder | 75.05% | Unfrozen ResNet101 |
| + Focal Loss | 74.40% | Better handling of hard examples |
| + BERT Embeddings | **77.04%** | Pre-trained text representations |
| Zero-shot (testB) | 65.45% | Transfer to new color-shape combinations |

### Performance Metrics (Best Model)
- **Accuracy**: 77.04%
- **Precision**: 76.88%
- **Recall**: 77.04%
- **F1-Score**: 76.77%

## 🔄 Model Variants

### 1. Base Model
- Frozen ResNet101 image encoder
- Custom transformer text encoder
- Cross-attention fusion

### 2. Fine-tuned Model
- Unfrozen ResNet101 for end-to-end training
- Improved visual feature learning

### 3. Focal Loss Model
- Addresses class imbalance in answer distribution
- Better handling of hard examples

### 4. BERT-Enhanced Model ⭐
- Pre-trained BERT embeddings initialization
- Best performing variant
- Superior semantic understanding

## 📊 Evaluation

The model is evaluated on multiple dimensions:

- **Quantitative Metrics**: Accuracy, Precision, Recall, F1-Score
- **Qualitative Analysis**: Visualization of correct and error cases
- **Robustness Testing**: Zero-shot evaluation on CLEVR-B
- **Error Analysis**: Common failure patterns in spatial reasoning and counting

### Common Error Patterns
- Complex multi-step spatial reasoning
- Fine-grained attribute differentiation
- Counting multiple object types simultaneously
- Out-of-distribution color-shape combinations

## 🎓 Academic Context

This project was developed as part of COL774: Machine Learning course assignment. The work demonstrates:
- Multi-modal deep learning techniques
- Transformer architectures for VQA
- Cross-attention mechanisms
- Transfer learning and domain adaptation

