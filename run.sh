#!/bin/bash

# run.sh - Bootstrap and run the Medical Chatbot locally

set -e

echo "=== Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS ==="
echo "Starting setup..."

# Step 1: Check for conda
if ! command -v conda &> /dev/null
then
    echo "❌ Conda could not be found. Please install Miniconda or Anaconda first."
    exit
fi

# Step 2: Create / activate conda env
if conda info --envs | grep -q "medichatbot"; then
    echo "✅ Conda env 'medichatbot' already exists"
else
    echo "➡️ Creating conda env 'medichatbot' with Python 3.10..."
    conda create -n medichatbot python=3.11 -y
fi

echo "➡️ Activating environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medichatbot

# Step 3: Install dependencies
echo "➡️ Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Ensure .env exists
if [ ! -f .env ]; then
    echo "⚠️ .env file not found!"
    echo "Please create a .env file with your PINECONE_API_KEY and OPENAI_API_KEY before running."
    exit 1
fi

# Step 5: Build Pinecone index if not already stored
if [ ! -f data/.pinecone_initialized ]; then
    echo "➡️ Storing embeddings into Pinecone..."
    python store_index.py
else
    echo "✅ Pinecone index already initialized"
fi

# Step 6: Run Flask app
echo "➡️ Starting Flask app on http://127.0.0.1:8080 ..."
python app.py
