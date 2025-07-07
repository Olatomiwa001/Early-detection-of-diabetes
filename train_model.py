# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (replace with your file path if needed)
df = pd.read_csv("diabetes.csv")  # Make sure it's in the same folder

# Features and label
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("✅ Model and features saved.")
