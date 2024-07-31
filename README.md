# Transformer Model Documentation

## Overview

This repository contains an implementation of a transformer-based language model inspired by Andrej Karpathy's minGPT project. The main goal of this project is to gain hands-on experience with coding up transformer models from research papers. The project includes two main scripts: `bigram.py` and `transformer.py`, and a sample dataset `input.txt`.

## Table of Contents

- [Files and Directories](#files-and-directories)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [bigram.py](#bigrampy)
  - [Overview](#overview)
  - [Usage](#usage)
- [transformer.py](#transformerpy)
  - [Overview](#overview-1)
  - [Usage](#usage-1)
- [Saving and Loading Model Weights](#saving-and-loading-model-weights)
- [Conclusion](#conclusion)

## Files and Directories

- `bigram.py`: Contains the implementation of a simple bigram language model.
- `transformer.py`: Contains the implementation of a transformer-based language model.
- `input.txt`: Sample dataset (Shakespeare's text) used for training the models.
- `model_weights.pth`: File to save the trained model weights.

## Dependencies

To run this project, you'll need the following libraries:

- Python 3.x
- PyTorch
- NumPy

You can install the required libraries using pip:

```bash
pip install torch numpy
```

## Getting Started
1. Clone the repository:

```bash
git clone https://github.com/yourusername/transformer-model.git
cd transformer-model
```
2. Ensure you have the required dependencies installed (see above).

3. Download the sample dataset (if not already present):

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## bigram.py
### Overview
`bigram.py` implements a simple bigram language model. This model predicts the next character in a sequence based on the previous character. It uses an embedding layer to represent characters and a simple neural network to make predictions.
### Usage
To train the bigram model, simply run the script:

```bash
python bigram.py
```

Key functions and classes in `bigram.py`:

- `encode()`: Encodes a string into a list of integers based on the character-to-integer mapping.
- `decode()`: Decodes a list of integers back into a string based on the integer-to-character mapping.
- `get_batch()`: Generates a batch of data for training.
- `estimate_loss()`: Estimates the training and validation loss.
- `BigramLanguageModel`: Defines the architecture of the bigram language model.
- Training loop: Trains the model and prints the loss at regular intervals.

## transformer.py
### Overview
`transformer.py` implements a transformer-based language model. This model leverages self-attention mechanisms to capture long-range dependencies in text. The architecture includes multi-head attention, feed-forward layers, and positional embeddings.

### Usage
To train the transformer model, simply run the script:
```bash
python transformer.py
```
Key functions and classes in `transformer.py`:

- `encode()`: Encodes a string into a list of integers based on the character-to-integer mapping.
- `decode()`: Decodes a list of integers back into a string based on the integer-to-character mapping.
- `get_batch()`: Generates a batch of data for training.
- `estimate_loss()`: Estimates the training and validation loss.
- `Head`: Implements a single head of self-attention.
- `MultiHeadAttention`: Implements multiple heads of attention in parallel.
- `FeedForward`: Defines a simple linear layer followed by a non-linearity.
- `Block`: Defines a transformer block consisting of multi-head attention and feed-forward layers.
- `GPTLanguageModel`: Defines the architecture of the transformer language model.
- Training loop: Trains the model and prints the loss at regular intervals.

## Saving and Loading Model Weights
The trained model weights are saved to `model_weights.pth`. You can load these weights into a model instance using:
```bash
model = GPTLanguageModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.to(device)
```

## Conclusion
This project provides an implementation of both a bigram and a transformer-based language model from scratch. It serves as a practical exercise in understanding and coding up modern NLP models based on research papers. Feel free to experiment with the code, modify the hyperparameters, and train on different datasets to further your understanding.