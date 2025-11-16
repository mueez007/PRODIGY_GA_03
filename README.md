# Task 3: Text Generation with Markov Chains

## Description
Generate Shakespearean-style text using Markov chains with the Markovify library.

## Setup
```bash
# Create virtual environment
python -m venv markov_env
source markov_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm