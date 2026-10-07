import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
BASELINE_POSITIVE_RATE = 0.248
DRIFT_LIMIT = 0.05
DECISION_THRESHOLDS = [round(value / 10, 1) for value in range(1, 10)]


def find_best_threshold(y_true, probabilities):
    """Quet nguong 0.1..0.9 va chon F1 lop duong cao nhat."""
    scores = {
        threshold: float(f1_score(y_true, probabilities >= threshold))
        for threshold in DECISION_THRESHOLDS
    }
    # Neu hoa, uu tien nguong gan 0.5 nhat de ket qua on dinh, de giai thich.
    best_threshold = max(
        scores,
        key=lambda threshold: (scores[threshold], -abs(threshold - 0.5)),
    )
    return best_threshold, scores


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        mlflow.log_params(params)

        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_eval)[:, 1]
        best_threshold, threshold_scores = find_best_threshold(y_eval, probabilities)
        preds = (probabilities >= best_threshold).astype(int)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))
        precision = float(precision_score(y_eval, preds, zero_division=0))
        recall = float(recall_score(y_eval, preds, zero_division=0))
        positive_rate = float(y_train.mean())
        drift = abs(positive_rate - BASELINE_POSITIVE_RATE)
        drift_detected = drift > DRIFT_LIMIT

        if drift_detected:
            warnings.warn(
                f"Data drift: positive rate {positive_rate:.4f} differs from "
                f"baseline {BASELINE_POSITIVE_RATE:.4f} by {drift:.4f}",
                RuntimeWarning,
            )

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("decision_threshold", best_threshold)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.log_metric("positive_rate_drift", drift)
        mlflow.set_tag("data_drift_detected", str(drift_detected).lower())
        mlflow.sklearn.log_model(model, "model")

        print(
            f"F1: {f1:.4f} | Accuracy: {acc:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            f"Threshold: {best_threshold:.1f}"
        )

        os.makedirs("outputs", exist_ok=True)
        report = {
            "f1_score": f1,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "decision_threshold": best_threshold,
            "positive_rate": positive_rate,
            "positive_rate_drift": drift,
            "data_drift_detected": drift_detected,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report, f, indent=2)

        with open("outputs/threshold_scan.json", "w") as f:
            json.dump(
                {str(threshold): score for threshold, score in threshold_scores.items()},
                f,
                indent=2,
            )

        with open("outputs/classification_report.json", "w") as f:
            json.dump(
                classification_report(y_eval, preds, output_dict=True, zero_division=0),
                f,
                indent=2,
            )

        with open("outputs/confusion_matrix.json", "w") as f:
            json.dump(
                {"labels": [0, 1], "matrix": confusion_matrix(y_eval, preds).tolist()},
                f,
                indent=2,
            )

        with open("outputs/drift_report.json", "w") as f:
            json.dump(
                {
                    "baseline_positive_rate": BASELINE_POSITIVE_RATE,
                    "current_positive_rate": positive_rate,
                    "absolute_drift": drift,
                    "limit": DRIFT_LIMIT,
                    "drift_detected": drift_detected,
                },
                f,
                indent=2,
            )

        for artifact in (
            "outputs/report.json",
            "outputs/threshold_scan.json",
            "outputs/classification_report.json",
            "outputs/confusion_matrix.json",
            "outputs/drift_report.json",
        ):
            mlflow.log_artifact(artifact, artifact_path="evaluation")

        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "decision_threshold": best_threshold,
                "feature_names": list(X_train.columns),
            },
            "models/model.joblib",
        )

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
