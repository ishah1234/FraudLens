import pandas as pd
import numpy as np
import faiss
import pickle
import requests
import json
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

print("Loading FraudLens data...")
embedder = pickle.load(open("embedder.pkl", "rb"))
texts = pickle.load(open("texts.pkl", "rb"))
index = faiss.read_index("vector_store.faiss")
flagged = pd.read_csv("flagged_transactions.csv")
print(f"Loaded {len(flagged)} flagged transactions")

def _get_high_risk(t=0.85):
    t = float(t)
    results = flagged[flagged['fraud_score'] >= t]
    if results.empty:
        return f"No transactions found above {t}"
    output = f"Found {len(results)} transactions above {t}:\n"
    for _, row in results.nlargest(5, 'fraud_score').iterrows():
        output += f"- {row['transaction_id']}: ${row['amount']}, {row['merchant_name']}, Score {row['fraud_score']:.2f}\n"
    return output

@tool
def get_transaction(txn_id: str) -> str:
    """Get details of a specific transaction by ID"""
    row = flagged[flagged['transaction_id'] == txn_id]
    if row.empty:
        return f"Transaction {txn_id} not found"
    row = row.iloc[0]
    return (f"Transaction {txn_id}: Amount ${row['amount']}, "
            f"Merchant {row['merchant_name']} ({row['merchant_category']}), "
            f"Card {row['card_type']}, Hour {row['hour']}:00, "
            f"Location {row['location']}, "
            f"Previous declines {row['previous_declines']}, "
            f"Fraud score {row['fraud_score']:.2f}, "
            f"Date {row['transaction_date']}")

@tool
def search_by_merchant(merchant_name: str) -> str:
    """Find all flagged transactions at a specific merchant"""
    results = flagged[flagged['merchant_name'].str.lower() == merchant_name.lower()]
    if results.empty:
        return f"No flagged transactions found for {merchant_name}"
    output = f"Found {len(results)} flagged transactions at {merchant_name}:\n"
    for _, row in results.iterrows():
        output += f"- {row['transaction_id']}: ${row['amount']}, Hour {row['hour']}:00, Score {row['fraud_score']:.2f}\n"
    return output

@tool
def search_by_location(location: str) -> str:
    """Find all flagged transactions at a specific location"""
    results = flagged[flagged['location'].str.lower() == location.lower()]
    if results.empty:
        return f"No flagged transactions found in {location}"
    output = f"Found {len(results)} flagged transactions in {location}:\n"
    for _, row in results.iterrows():
        output += f"- {row['transaction_id']}: ${row['amount']}, {row['merchant_name']}, Hour {row['hour']}:00, Score {row['fraud_score']:.2f}\n"
    return output

@tool
def search_similar_transactions(query: str) -> str:
    """Search for transactions similar to a description using semantic search"""
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector.astype(np.float32), k=3)
    results = [texts[i] for i in indices[0]]
    return "Similar transactions found:\n" + "\n".join(results)

@tool
def get_high_risk_transactions(threshold: str = "0.85") -> str:
    """Get all transactions above a certain fraud score threshold"""
    try:
        t = float(str(threshold).strip()) if threshold and str(threshold) not in ["None", "", "null"] else 0.85
        if t > 1.0:
            t = 0.85
    except:
        t = 0.85
    return _get_high_risk(t)

class AgentState(TypedDict):
    goal: str
    steps: Annotated[list, operator.add]
    findings: Annotated[list, operator.add]
    final_report: str
    done: bool

tools = [get_transaction, search_by_merchant, search_by_location, search_similar_transactions, get_high_risk_transactions]
tools_map = {t.name: t for t in tools}

def planner(state: AgentState) -> AgentState:
    goal = state['goal']
    prompt = f"""You are a fraud investigation agent at Aurus payments company.

Your goal: {goal}

You have these tools:
- get_transaction(txn_id): Get details of a specific transaction
- search_by_merchant(merchant_name): Find flagged transactions at a merchant
- search_by_location(location): Find flagged transactions in a location
- search_similar_transactions(query): Semantic search for similar transactions
- get_high_risk_transactions(threshold): Get highest risk transactions, always pass a decimal like "0.85"

Plan exactly 3 steps. Respond ONLY with a JSON array:
[
  {{"tool": "get_transaction", "input": "TXN_0238"}},
  {{"tool": "search_by_merchant", "input": "Shell"}},
  {{"tool": "get_high_risk_transactions", "input": "0.85"}}
]
IMPORTANT: For get_high_risk_transactions always pass a decimal between 0 and 1 like "0.85". Never null or None.
Only the JSON array. Nothing else."""

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    raw = response.json()['response'].strip()
    start = raw.find('[')
    end = raw.rfind(']') + 1
    steps = json.loads(raw[start:end])

    print(f"\nAgent planned {len(steps)} steps:")
    for i, s in enumerate(steps):
        print(f"  {i+1}. {s['tool']}({s['input']})")

    return {"steps": steps, "findings": [], "done": False, "final_report": ""}

def executor(state: AgentState) -> AgentState:
    steps = state['steps']
    findings = []

    for step in steps:
        tool_name = step['tool']
        tool_input = step['input']
        print(f"\nExecuting: {tool_name}({tool_input})")

        if tool_name not in tools_map:
            findings.append(f"[{tool_name}]: Tool not found")
            continue

        try:
            if tool_name == "get_high_risk_transactions":
                try:
                    t = float(str(tool_input).strip()) if tool_input and str(tool_input) not in ["None", "", "null"] else 0.85
                    if t > 1.0:
                        t = 0.85
                except:
                    t = 0.85
                result = _get_high_risk(t)
            else:
                result = tools_map[tool_name].invoke(tool_input)

            findings.append(f"[{tool_name}]: {result}")
            print(f"Result: {result[:100]}...")

        except Exception as e:
            findings.append(f"[{tool_name}]: Error - {str(e)}")
            print(f"Error: {str(e)}")

    return {"findings": findings}

def reporter(state: AgentState) -> AgentState:
    goal = state['goal']
    findings = state['findings']

    prompt = f"""You are a senior fraud analyst at Aurus payments company.

Investigation Goal: {goal}

Findings:
{chr(10).join(findings)}

Write a professional fraud investigation report with:
1. Executive Summary (2-3 sentences)
2. Key Risk Indicators
3. Most suspicious transactions with reasons
4. Recommended actions

Be specific with transaction IDs and amounts."""

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    report = response.json()['response']
    print(f"\n{'='*50}\nINVESTIGATION REPORT\n{'='*50}")
    print(report)
    return {"final_report": report, "done": True}

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("reporter", reporter)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reporter")
workflow.add_edge("reporter", END)
agent = workflow.compile()

if __name__ == "__main__":
    goals = [
        "Investigate TXN_0238 and find related suspicious activity",
        "Find the highest risk transactions and explain why they are dangerous",
    ]
    for goal in goals:
        print(f"\n{'='*50}\nGOAL: {goal}\n{'='*50}")
        result = agent.invoke({
            "goal": goal,
            "steps": [],
            "findings": [],
            "final_report": "",
            "done": False
        })
