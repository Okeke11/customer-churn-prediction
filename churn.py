import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. LOAD DATA
# ==========================================
# Make sure the CSV file is in the same folder as this script
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset Loaded Successfully!")
except FileNotFoundError:
    print("Error: File not found. Please download the CSV from Kaggle and place it in this folder.")
    exit()

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================

# A. Handle 'TotalCharges'
# The dataset has some blank strings " " which act as errors. We force them to NaN, then drop them.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Drop rows with missing TotalCharges only
df = df[~df['TotalCharges'].isna()].copy()  # Remove rows with missing TotalCharges

# B. Drop irrelevant columns
# 'customerID' is unique to every person and has no predictive value
df.drop('customerID', axis=1, inplace=True)

# C. Encode Categorical Data
# Strip whitespace in object columns
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

# Encode target 'Churn' explicitly and exclude it from feature encoding
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Determine object feature columns (excluding target)
obj_feature_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'Churn']

# Separate binary and multi-category columns
binary_cols = [c for c in obj_feature_cols if df[c].nunique() == 2]
multi_cols = [c for c in obj_feature_cols if df[c].nunique() > 2]

# Encode binary columns with a new LabelEncoder per column
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# One-Hot Encode multi-category columns
if len(multi_cols) > 0:
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

print("Data Preprocessing Complete.")
print(f"Final Data Shape: {df.shape}")

# ==========================================
# 3. SPLITTING THE DATA
# ==========================================
# X is everything EXCEPT Churn, y is ONLY Churn
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATION
# ==========================================
y_pred = model.predict(X_test)

print("\n--- RESULTS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Visualizing Feature Importance (What causes churn?)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Factors Driving Churn')
plt.tight_layout()
plt.savefig('feature_importance_top10.png')
plt.show()

import joblib

# Save the model to a file named 'churn_model.pkl'
joblib.dump(model, 'churn_model.pkl')

print("Success! Model saved as 'churn_model.pkl'")