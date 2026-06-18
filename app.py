import pandas as pd
import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import agent 

# Load everything
print("Loading models and data...")
embedder = pickle.load(open("embedder.pkl", "rb"))
texts = pickle.load(open("texts.pkl", "rb"))
index = faiss.read_index("vector_store.faiss")
flagged = pd.read_csv("flagged_transactions.csv")
print(f"Loaded {index.ntotal} transactions in vector store")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

def retrieve(query, k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(
        query_vector.astype(np.float32), k=k
    )
    results = [texts[i] for i in indices[0]]
    return results

def ask_ollama(query, context):
    prompt = f"""You are a fraud analyst at a payments company called Aurus.

You have been given the following flagged transactions as context:

{chr(10).join([f"- {t}" for t in context])}

Based on these transactions, answer this question:
{query}

Be specific, mention transaction IDs, amounts, and explain 
why each transaction looks suspicious. Keep it concise.
"""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()['response']

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/ask")
def ask(query: Query):
    context = retrieve(query.question, k=3)
    answer = ask_ollama(query.question, context)
    return {
        "question": query.question,
        "retrieved_transactions": context,
        "answer": answer
    }
@app.post("/investigate")
def investigate(query: Query):
    try:
        result = agent.invoke({
            "goal": query.question,
            "steps": [],
            "findings": [],
            "final_report": "",
            "done": False
        })
        return {
            "goal": query.question,
            "findings": result["findings"],
            "report": result["final_report"]
        }
    except Exception as e:
        return {"error": str(e), "goal": query.question, "findings": [], "report": ""}

@app.get("/flagged")
def get_flagged():
    top = flagged.nlargest(10, 'fraud_score')
    return top[['transaction_id', 'amount', 'merchant_name', 
                 'hour', 'location', 'previous_declines', 
                 'fraud_score']].to_dict(orient='records')