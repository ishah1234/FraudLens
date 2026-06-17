import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

merchants = [
    ("Whole Foods", "grocery"),
    ("Shell", "petro"),
    ("Best Buy", "electronics"),
    ("Starbucks", "dining"),
    ("Amazon", "ecommerce"),
    ("Walmart", "grocery"),
    ("Target", "retail"),
    ("Exxon", "petro")
]

locations = ["Boston MA", "New York NY", "Chicago IL", "Phoenix AZ", "Houston TX"]
cards = ["Visa", "Mastercard", "Amex"]

rows = []
for i in range(1000):
    merchant = random.choice(merchants)
    is_fraud = 1 if random.random() < 0.05 else 0

    if is_fraud:
        # Fraud is mostly late night but not always
        hour = random.choices(
            [random.randint(0, 4), random.randint(8, 22)],
            weights=[70, 30]  # 70% late night, 30% normal hours
        )[0]

        # Fraud is mostly high amount but not always
        amount = random.choices(
            [round(random.uniform(500, 2000), 2), round(random.uniform(5, 300), 2)],
            weights=[65, 35]  # 65% high, 35% normal amount
        )[0]

        # Fraud usually has declines but not always
        declines = random.choices(
            [random.randint(1, 3), 0],
            weights=[60, 40]  # 60% has declines, 40% doesn't
        )[0]

    else:
        # Normal transactions occasionally happen late night
        hour = random.choices(
            [random.randint(8, 22), random.randint(0, 4)],
            weights=[90, 10]  # 90% normal hours, 10% late night
        )[0]

        # Normal transactions occasionally are high value
        amount = random.choices(
            [round(random.uniform(5, 300), 2), round(random.uniform(500, 2000), 2)],
            weights=[85, 15]  # 85% normal, 15% high value
        )[0]

        # Normal transactions rarely have declines
        declines = random.choices(
            [0, random.randint(1, 2)],
            weights=[92, 8]  # 92% no declines, 8% has declines
        )[0]

    date = datetime.now() - timedelta(days=random.randint(0, 30))

    rows.append({
        "transaction_id": f"TXN_{i:04d}",
        "amount": amount,
        "merchant_name": merchant[0],
        "merchant_category": merchant[1],
        "card_type": random.choice(cards),
        "hour": hour,
        "previous_declines": declines,
        "location": random.choice(locations),
        "transaction_date": date.strftime("%Y-%m-%d"),
        "is_fraud": is_fraud
    })

df = pd.DataFrame(rows)
df.to_csv("transactions.csv", index=False)
print(f"Generated {len(df)} transactions")
print(f"Fraud cases: {df['is_fraud'].sum()}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
print("\nSample fraud transactions:")
print(df[df['is_fraud']==1][['amount','hour','previous_declines']].head())
print("\nSample normal transactions:")
print(df[df['is_fraud']==0][['amount','hour','previous_declines']].head())