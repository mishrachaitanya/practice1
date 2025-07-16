import os
import joblib
import pandas as pd
import pytest
from train import clf, X_test, y_test, acc

# === Test 1: Check data file exists ===
def test_data_file_exists():
    assert os.path.exists("data/train.csv"), "Data file does not exist."

    # === Test 2: Check model pipeline ===
def test_pipeline_object():
    assert clf is not None, "Model pipeline is not created."
    assert hasattr(clf, "predict"), "Pipeline does not have predict method."

                    # === Test 3: Model accuracy threshold ===
def test_model_accuracy():
    assert acc > 0.6, f"Model accuracy is too low: {acc}"

                            # === Test 4: Model file saved ===
def test_model_file():
    assert os.path.exists("model/titanic_model.joblib"), "Model file not saved."
                              # === Test 5: Predictions shape ===
def test_prediction_shape():
    preds = clf.predict(X_test)
    assert len(preds) == len(y_test), "Prediction length does not match test data."

