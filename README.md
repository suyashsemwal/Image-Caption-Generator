# Image Caption Generator

A deep learning-based image caption generator that uses VGG16 for image feature extraction and LSTM for generating natural-language captions.

## Live Demo

Try the deployed application: [Image Caption Generator](https://image-caption-generator-ai.streamlit.app)

## Project Overview

This project takes an input image and automatically generates a descriptive caption for it.

The model was trained using the Flickr8k dataset and uses a CNN-LSTM architecture:

- **VGG16** extracts visual features from the image.
- **LSTM** generates the caption word by word.
- **Tokenizer** converts words into numerical sequences.
- **Streamlit** provides the web interface.
