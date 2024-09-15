# AI Chatbot

This is a simple AI-based chatbot designed using Python, neural networks, and basic machine learning techniques. The bot is trained to understand and respond to various user queries.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Overview

The chatbot is designed to respond to a set of pre-defined intents using a neural network model built with Keras. The training data is organized in JSON format, and the bot can classify intents and generate responses based on those intents. This chatbot uses natural language processing (NLP) techniques to tokenize and vectorize input text before feeding it to the model.

## Project Structure

- `.gitattributes`: Defines attributes for the Git version control system.
- `Ai=chatbot.py`: The main file that contains the logic to run the AI-based chatbot.
- `Chatmodel.h5`: The trained model saved in HDF5 format.
- `README.md`: This file, containing documentation about the project.
- `chatbot.py`: The Python script to interact with the chatbot.
- `classes.pkl`: A pickle file storing the classes (intents) after preprocessing.
- `qs.json`: A JSON file that contains the training data in the form of intents and responses.
- `words.pkl`: A pickle file storing tokenized words after preprocessing.

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- TensorFlow or Keras
- NLTK (Natural Language Toolkit)
- NumPy
- Flask (if you are deploying the chatbot as a web application)

### Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/PraneshxxTechy/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**:
    Make sure you have a virtual environment set up, then install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

    If there's no `requirements.txt`, manually install:
    ```bash
    pip install tensorflow keras nltk numpy flask
    ```

3. **Download NLTK corpus**:
    The bot requires NLTK’s `punkt` tokenizer. Run this in Python:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

To run the chatbot, execute the `chatbot.py` script:
```bash
python chatbot.py
```

If using `Ai=chatbot.py` to directly interact with the AI, you can run that file similarly:
```bash
python Ai=chatbot.py
```

Once running, the bot will load the model and start interacting based on the user's inputs.

## Model Training

The chatbot uses a neural network that has been trained on data in `qs.json`. The `Chatmodel.h5` file stores the trained model. If you want to retrain the model or use new intents, follow these steps:

1. Modify or add new intents in the `qs.json` file.
2. Train the model by executing the training script (if it exists, or you can modify `chatbot.py` for retraining).
3. After training, the model will be saved in `Chatmodel.h5`.

Here’s how you can retrain the model if necessary:
```bash
python train_chatbot.py  # Adjust the script name if different
```

