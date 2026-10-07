import os
import json
import numpy as np
import pandas as pd
import joblib
from src.train import DECISION_THRESHOLDS, find_best_threshold, train


FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Adult de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi cloud storage hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))

    y = rng.integers(0, 2, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path, monkeypatch):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)

    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path, monkeypatch):
    """Kiem tra file outputs/report.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.json")
    with open("outputs/report.json") as f:
        report = json.load(f)
    assert "f1_score" in report
    assert "accuracy" in report
    assert "precision" in report
    assert "recall" in report
    assert report["decision_threshold"] in DECISION_THRESHOLDS
    assert "data_drift_detected" in report

    for artifact in (
        "threshold_scan.json",
        "classification_report.json",
        "confusion_matrix.json",
        "drift_report.json",
    ):
        assert os.path.exists(f"outputs/{artifact}")


def test_model_file_created(tmp_path, monkeypatch):
    """Kiem tra file models/model.joblib duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.joblib")
    artifact = joblib.load("models/model.joblib")
    assert "model" in artifact
    assert artifact["decision_threshold"] in DECISION_THRESHOLDS
    assert artifact["feature_names"] == FEATURE_NAMES


def test_find_best_threshold_uses_positive_class_f1():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.35, 0.45, 0.95])
    threshold, scores = find_best_threshold(y_true, probabilities)

    assert threshold == 0.4
    assert scores[threshold] == 1.0
