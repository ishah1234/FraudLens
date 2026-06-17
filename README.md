# FraudLens

An AI-powered fraud intelligence system that lets analysts ask
plain English questions over flagged payment transactions.

Built with Random Forest, HuggingFace embeddings, FAISS,
RAG pipeline, and Llama 3 running locally via Ollama.

## Tech Stack

- scikit-learn — fraud detection model
- HuggingFace sentence-transformers — semantic embeddings
- FAISS — vector store for similarity search
- Ollama + Llama 3 — local LLM for explanation generation
- FastAPI + Uvicorn — backend API
- Vanilla HTML/CSS/JS — frontend

## Setup

### 1. Clone the repo

git clone https://github.com/ishah1234/fraudlens
cd fraudlens

### 2. Install dependencies

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3. Install Ollama

Download from https://ollama.com
Then run:
ollama pull llama3

### 4. Generate data and build models

python generate_data.py
python train_model.py
python rag.py

### 5. Start Ollama in a separate terminal

ollama serve

### 6. Run the app

uvicorn app:app --reload

Open http://localhost:8000

## How it works

1. Random Forest ML model scores 1000 transactions for fraud
2. Flags suspicious ones with fraud score above 0.5
3. HuggingFace converts flagged transactions to embeddings
4. FAISS stores embeddings for semantic search
5. You ask a question in plain English
6. FAISS retrieves the 3 most relevant transactions
7. Llama 3 explains them like a fraud analyst
