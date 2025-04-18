import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dtreeviz import model  # Visualization
import dtreeviz
import pickle  # If querying the oracle

# 🟠 Step 1: Load UCI Heart Disease Dataset
url = "/home/piyush/umc203/heart+disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                "oldpeak", "slope", "ca", "thal", "target"]

# Load dataset
df = pd.read_csv(url, names=column_names)

# 🟠 Step 2: Data Cleanup
# Replace '?' with NaN and drop rows with missing values
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert categorical columns to numeric
df = df.astype(float)

# Convert target variable: 
# UCI dataset has 0 (No Disease) and [1,2,3,4] (Disease)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Split data into features (X) and target (y)
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟠 Step 3: Query Oracle for Hyperparameters
def query_oracle():
    # Example: Replace this with actual Oracle query
    return {"criterion": "gini", "splitter": "best", "max_depth": 5}

oracle_params = query_oracle()
criterion = oracle_params["criterion"]
splitter = oracle_params["splitter"]
max_depth = oracle_params["max_depth"]

# 🟠 Step 4: Train Decision Tree Model
classifier = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=42)
classifier.fit(X_train, y_train)

# 🟠 Step 5: Evaluate Model Performance
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print Classification Report
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# Save evaluation metrics
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
print("\n🔹 Model Performance:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# 🟠 Step 6: Visualize Decision Tree
viz = model(classifier, X_train, y_train, target_name="Heart Disease",
               feature_names=list(X.columns), class_names=["No Disease", "Disease"])

# Save the decision tree visualization as an SVG file
# viz.export_to_file("decision_tree.svg")  # ✅ Correct method
viz.fontname = "DejaVu Sans"  # Set font for better visualization

viz.view()


