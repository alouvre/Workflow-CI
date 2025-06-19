import argparse
import json
from pathlib import Path

import mlflow
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from mlflow.models.signature import infer_signature

# ----------------------------------
# 🔧 Setup MLflow tracking lokal
# ----------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Dropout_Prediction_Submission_NoTuning")


# ----------------------------------
# 📂 Parse arguments
# ----------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_student_cleaned.csv', help='Path ke dataset')
    return parser.parse_args()


# ----------------------------------
# 📂 Load dan split data
# ----------------------------------
def load_data(path="data_student_cleaned.csv"):
    df = pd.read_csv(path)
    X = df.drop("Status", axis=1)
    y = df["Status"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def log_classification_report(y_true, y_pred, filename):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(filename)


def log_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)


def log_estimator_html(model, filename):
    with open(filename, "w") as f:
        f.write("<html><body><h2>Best Estimator</h2><pre>")
        f.write(str(model))
        f.write("</pre></body></html>")
    mlflow.log_artifact(filename)


def main(data_path):
    X_train, X_test, y_train, y_test = load_data(data_path)

    model_name = "XGBoost"
    model = XGBClassifier(eval_metric='logloss', n_jobs=1)

    with mlflow.start_run(run_name="Model_NoTuning") as run:
        print(f"🔍 Training model: {model_name}...")

        mlflow.autolog(disable=True)  # Nonaktifkan autolog agar tidak bentrok saat log manual

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # 📁 Buat folder artifacts manual
        run_id = run.info.run_id
        artifact_dir = Path("mlartifacts") / run_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Logging artifacts
        log_classification_report(y_test, y_pred, artifact_dir / f"{model_name}_metric_info.json")
        # log_confusion_matrix(y_test, y_pred, artifact_dir / f"{model_name}_cm.png")
        log_estimator_html(pipeline, artifact_dir / f"{model_name}_estimator.html")

        # Simpan model .pkl manual
        model_path = artifact_dir / f"{model_name}_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        # Logging model secara lengkap via MLflow
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"{model_name}_mlflow_model",
            signature=signature
        )

        print(f"✅ Model {model_name} selesai dilatih dan dicatat ke MLflow.")

    print("🎉 Proses selesai tanpa tuning.")


# ----------------------------------
# ▶️ Entry point
# ----------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args.data_path)
