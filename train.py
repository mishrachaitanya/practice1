import pandas as pd
import numpy as np
import os
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow
import mlflow.sklearn

# === Train function ===
def train_and_evaluate(model_path, metrics_path):
    # Load Data
    data_path = "data/train.csv"
    df = pd.read_csv(data_path)

    # Drop unnecessary columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Save metrics in given format
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df = pd.DataFrame([{
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

    # MLflow logging (local tracking)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Titanic-Experiment")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(clf, "model")
        mlflow.log_artifact(metrics_path)

# === Main entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model/titanic_model.joblib")
    parser.add_argument("--metrics-path", type=str, default="metrics/metrics.csv")
    args = parser.parse_args()

    train_and_evaluate(args.model_path, args.metrics_path)

