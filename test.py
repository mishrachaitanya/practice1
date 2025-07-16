import os
import pandas as pd
from train import train_and_evaluate

MODEL_PATH = "model/test_model.joblib"
METRICS_PATH = "metrics/test_metrics.csv"

def setup_module(module):
    # Train model before running tests
    train_and_evaluate(MODEL_PATH, METRICS_PATH)

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Model file was not created."

def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH), "Metrics file was not created."

def test_metrics_format():
    df = pd.read_csv(METRICS_PATH)
    expected_cols = {"accuracy", "precision", "recall", "f1"}
    assert set(df.columns) == expected_cols, f"Metrics CSV columns are wrong: {df.columns}"

def test_accuracy_threshold():
    df = pd.read_csv(METRICS_PATH)
    acc = df["accuracy"].iloc[0]
    assert acc > 0.6, f"Accuracy too low: {acc}"

