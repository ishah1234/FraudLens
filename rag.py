from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import pickle

# Load the trained model and data
df = pd.read_csv("transactions.csv")
model_ml = pickle.load(open("model.pkl", "rb"))
le_merchant = pickle.load(open("le_merchant.pkl", "rb"))
le_card = pickle.load(open("le_card.pkl", "rb"))
le_location = pickle.load(open("le_location.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Step 1: Score all transactions with the ML model
print("Scoring all transactions...")

def preprocess(row):
    data = pd.DataFrame([{
        'amount': row['amount'],
        'hour': row['hour'],
        'previous_declines': row['previous_declines'],
        'merchant_category_enc': le_merchant.transform([row['merchant_category']])[0],
        'card_type_enc': le_card.transform([row['card_type']])[0],
        'location_enc': le_location.transform([row['location']])[0]
    }])
    return data[features].values[0]

df['fraud_score'] = df.apply(
    lambda row: model_ml.predict_proba([preprocess(row)])[0][1], axis=1
)
df['predicted_fraud'] = (df['fraud_score'] > 0.5).astype(int)

flagged = df[df['predicted_fraud'] == 1].copy()
print(f"Flagged {len(flagged)} transactions as suspicious")

# Step 2: Convert each flagged transaction to a sentence
# This is what gets embedded and stored in the vector store
def transaction_to_text(row):
    return (
        f"Transaction {row['transaction_id']}: "
        f"Amount ${row['amount']}, "
        f"Merchant {row['merchant_name']} ({row['merchant_category']}), "
        f"Card {row['card_type']}, "
        f"Hour {row['hour']}:00, "
        f"Location {row['location']}, "
        f"Previous declines {row['previous_declines']}, "
        f"Fraud score {row['fraud_score']:.2f}, "
        f"Date {row['transaction_date']}"
    )

flagged['text'] = flagged.apply(transaction_to_text, axis=1)
print("\nSample transaction text:")
print(flagged['text'].iloc[0])

# Step 3: Load HuggingFace embedding model
# This converts text into numbers (vectors) that capture meaning
print("\nLoading HuggingFace embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Embed all flagged transactions
print("Embedding flagged transactions...")
texts = flagged['text'].tolist()
embeddings = embedder.encode(texts, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")
# Each transaction is now a vector of 384 numbers

# Step 5: Store embeddings in FAISS vector store
print("\nBuilding FAISS vector store...")
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)  # L2 = euclidean distance
index.add(embeddings.astype(np.float32))
print(f"Vector store has {index.ntotal} transactions")

# Step 6: Save everything for the app
faiss.write_index(index, "vector_store.faiss")
flagged.to_csv("flagged_transactions.csv", index=False)
pickle.dump(texts, open("texts.pkl", "wb"))
pickle.dump(embedder, open("embedder.pkl", "wb"))

print("\nRAG pipeline ready!")
print(f"Vector store saved with {index.ntotal} flagged transactions")

# Step 7: Test retrieval
print("\nTesting retrieval...")
test_query = "high amount transaction late in the morning"
query_embedding = embedder.encode([test_query])
distances, indices = index.search(query_embedding.astype(np.float32), k=3)

print(f"\nTop 3 transactions matching '{test_query}':")
for i, idx in enumerate(indices[0]):
    print(f"\n{i+1}. {texts[idx]}")