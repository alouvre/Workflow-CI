import os
import json
from pathlib import Path
import joblib
import mlflow
import pandas as pd

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from mlflow.models.signature import infer_signature

# ----------------------------------
# 🔧 Setup MLflow tracking lokal
# ----------------------------------
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:MLproject/mlruns"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Dropout_Prediction_Submission")
mlflow.autolog(disable=True)  # Nonaktifkan autolog agar tidak bentrok saat log manual


# ----------------------------------
# 📂 Load dan split data
# ----------------------------------
def load_data(path="data_student_cleaned.csv"):
    df = pd.read_csv(path)
    X = df.drop("Status", axis=1)
    y = df["Status"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def log_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(str(Path(save_path)))


def log_estimator_html(model, save_path):
    with open(save_path, "w") as f:
        f.write("<html><body><h2>Best Estimator</h2><pre>")
        f.write(str(model))
        f.write("</pre></body></html>")
    mlflow.log_artifact(str(Path(save_path)))


# ----------------------------------
# ▶️ Main Training
# ----------------------------------
def main():
    X_train, X_test, y_train, y_test = load_data()
    input_example = X_train[0:5]

    model_name = "XGBoost"
    model = XGBClassifier(eval_metric='logloss', n_jobs=1)

    with mlflow.start_run() as run:
        print(f"🔍 Training model: {model_name}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Logging metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metrics({
                f"test_{model_name}_accuracy_score": acc,
                f"test_{model_name}_precision_score": prec,
                f"test_{model_name}_recall_score": rec,
                f"test_{model_name}_f1_score": f1
            })

        # 📁 Buat folder artifacts manual
        base_dir = Path(__file__).resolve().parent
        artifact_dir = base_dir / "mlartifacts" / run.info.run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Logging artifacts
        log_classification_report(y_test, y_pred, artifact_dir / f"{model_name}_metric_info.json")
        log_estimator_html(pipeline, artifact_dir / f"{model_name}_estimator.html")

        # Simpan model .pkl manual
        model_path = artifact_dir / f"{model_name}_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        # 🔁 Simpan model dengan struktur MLflow lengkap
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"best_{model_name}_model",  # ini akan muncul sebagai folder di artifacts
            signature=signature,
            input_example=input_example
        )

        print(f"✅ Model {model_name} selesai dilatih dan dicatat ke MLflow.")
        print(f"🔍 Akurasi: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}  | F1-Score: {f1:.4f}")

    print("🎉 Proses selesai tanpa tuning.")


# ----------------------------------
# ▶️ Entry point
# ----------------------------------
if __name__ == "__main__":
    main()
