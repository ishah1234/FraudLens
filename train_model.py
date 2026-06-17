import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import numpy as np

# Load data
df = pd.read_csv("transactions.csv")
print("Data loaded:", df.shape)

# Encode categorical columns
# ML models only understand numbers, not text like "grocery" or "Visa"
le_merchant = LabelEncoder()
le_card = LabelEncoder()
le_location = LabelEncoder()

df['merchant_category_enc'] = le_merchant.fit_transform(df['merchant_category'])
df['card_type_enc'] = le_card.fit_transform(df['card_type'])
df['location_enc'] = le_location.fit_transform(df['location'])

# Select features for the model
features = ['amount', 'hour', 'previous_declines', 
            'merchant_category_enc', 'card_type_enc', 'location_enc']

X = df[features]
y = df['is_fraud']

print(f"Features: {features}")
print(f"Fraud cases: {y.sum()} out of {len(y)}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
# class_weight='balanced' handles the imbalance (42 fraud vs 958 normal)
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
print("\nModel trained!")

# Evaluate
predictions = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, predictions))

# Feature importance — which features matter most?
importance = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

# Save everything we need later
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_merchant, open("le_merchant.pkl", "wb"))
pickle.dump(le_card, open("le_card.pkl", "wb"))
pickle.dump(le_location, open("le_location.pkl", "wb"))
pickle.dump(features, open("features.pkl", "wb"))

print("\nModel and encoders saved!")