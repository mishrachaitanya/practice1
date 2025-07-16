import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# === Load Data ===
data_path = "data/train.csv"  # Assume data is in data/ folder
df = pd.read_csv(data_path)

# === Basic Preprocessing ===
# Drop columns that leak data or aren't useful
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Split features/target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# === Define preprocessing pipelines ===
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
            ])

categorical_features = ["Pclass", "Sex", "Embarked"]
categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
            ])

# === Create final model pipeline ===
clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
            ])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# === Save model ===
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/titanic_model.joblib")
print("Model saved to model/titanic_model.joblib")

