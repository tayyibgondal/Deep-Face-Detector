# Neural Machine Translation: Spanish to English

This project demonstrates how to train a sequence-to-sequence (seq2seq) model for Spanish-to-English translation roughly based on [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025v5) (Luong et al., 2015).

![Encoder/decoder connected by attention](https://www.tensorflow.org/images/tutorials/transformer/RNN%2Battention-words-spa.png)

## Overview

The project provides a step-by-step guide to train an encoder-decoder model with attention for Spanish-to-English translation. The architecture is based on the research paper mentioned above.

## Dependencies

To run this notebook, you'll need the following dependencies:

- Python (version X.X.X)
- TensorFlow (version X.X.X)
- Other necessary libraries

## Dataset

The tutorial uses a dataset of Spanish-English sentence pairs. However, the dataset itself is not provided in this tutorial. You'll need to obtain a suitable dataset or use your own parallel corpus of Spanish-English sentences.

## Training

The training process involves the following steps:

1. Data preprocessing: Preprocess the dataset by tokenizing the sentences, creating vocabulary, and splitting into train and validation sets.
2. Model architecture: Build the seq2seq model with attention using the TensorFlow framework.
3. Training loop: Train the model on the preprocessed dataset, monitor the loss, and save the checkpoints.
4. Evaluation: Evaluate the model's performance on the validation set using appropriate metrics.

## Inference

Once the model is trained, you can perform inference using the following steps:

1. Preprocess the input sentence by tokenizing and converting it into the model's input format.
2. Load the trained model checkpoints.
3. Pass the preprocessed input through the model and generate the translated output sentence.

## Acknowledgments

This project is based on the TensorFlow tutorial available at [TensorFlow Neural Machine Translation Tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention).


