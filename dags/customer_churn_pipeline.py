
"""Customer Churn MLOps DAG.

S3 -> SQS listener -> Airflow -> Training -> MLflow -> Registry.
Model files are stored in both S3 and MLflow run artifacts.
"""

from datetime import datetime, timezone
from io import StringIO
import json
import logging
import os
import pickle
import sys
import tempfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import quote, urlparse

import boto3
import joblib
import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
MLFLOW_EXPERIMENT_ID = "2"
REGISTERED_MODEL_NAME = "CustomerChurnModel"
BUCKET = "mlops-dvc-sunil-20261007"
DEFAULT_KEY = "datasets/customer_data.csv"
FEATURES = ["age", "income", "balance"]
REQUIRED = ["customer_id", *FEATURES, "left_bank"]


def mlflow_api_request(endpoint, payload):
    """Send tracking and registry requests to MLflow without its SDK."""
    request = Request(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_mlflow_artifact(artifact_uri, local_file, artifact_path="model"):
    """Upload model file bytes through MLflow's artifact HTTP endpoint."""
    parsed = urlparse(artifact_uri)

    if parsed.scheme != "mlflow-artifacts":
        raise AirflowException(f"Unexpected artifact URI: {artifact_uri}")

    # Use the actual artifact location supplied by MLflow.
    root = f"{parsed.netloc}{parsed.path}".strip("/")
    destination = f"{root}/{artifact_path}/{os.path.basename(local_file)}"

    url = (
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow-artifacts/artifacts/"
        f"{quote(destination, safe='/')}"
    )

    with open(local_file, "rb") as handle:
        request = Request(
            url,
            data=handle.read(),
            headers={"Content-Type": "application/octet-stream"},
            method="PUT",
        )

    with urlopen(request, timeout=120) as response:
        logging.info(
            "MLflow artifact uploaded: %s (HTTP %s)",
            destination,
            response.status,
        )


def load_customer_data(**context):
    """Read the CSV file identified by the incoming S3 event."""
    conf = context["dag_run"].conf or {}
    bucket = conf.get("bucket", BUCKET)
    key = conf.get("key", DEFAULT_KEY)

    logging.info("Reading s3://%s/%s", bucket, key)

    response = boto3.client("s3", region_name="us-east-1").get_object(
        Bucket=bucket, Key=key
    )

    df = pd.read_csv(StringIO(response["Body"].read().decode("utf-8")))
    logging.info("Loaded %d records", len(df))
    return df.to_json(orient="split")


def validate_data(**context):
    """Check that training data is complete and usable."""
    ti = context["ti"]
    df = pd.read_json(
        StringIO(ti.xcom_pull(task_ids="load_customer_data")),
        orient="split",
    )

    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise AirflowException(f"Missing columns: {missing}")

    if df.empty:
        raise AirflowException("Dataset is empty")

    if df[REQUIRED].isnull().any().any():
        raise AirflowException("Missing values detected")

    if df["customer_id"].duplicated().any():
        raise AirflowException("Duplicate customer IDs detected")

    if not df["left_bank"].isin(["Yes", "No"]).all():
        raise AirflowException("Invalid churn labels")

    if df["left_bank"].nunique() != 2 or df["left_bank"].value_counts().min() < 2:
        raise AirflowException("At least two examples per class are required")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="raise")

    logging.info("Data validation passed")
    return df.to_json(orient="split")


def preprocess_data(**context):
    """Prepare the features and split data into training and testing sets."""
    df = pd.read_json(
        StringIO(context["ti"].xcom_pull(task_ids="validate_data")),
        orient="split",
    )

    X = df[FEATURES].astype(float)
    y = df["left_bank"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=42,
        stratify=y,
    )

    logging.info("Training: %d; testing: %d", len(X_train), len(X_test))

    return {
        "X_train": X_train.values.tolist(),
        "X_test": X_test.values.tolist(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
    }


def train_model(**context):
    """Train, save to S3, and upload actual artifacts to MLflow."""
    data = context["ti"].xcom_pull(task_ids="preprocess_data")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(data["X_train"], data["y_train"])

    safe_run_id = context["run_id"].replace("/", "_").replace(":", "_")
    model_key = f"models/customer-churn/{safe_run_id}/model.joblib"

    # Create the MLflow run BEFORE the temporary model files are deleted.
    run = mlflow_api_request(
        "runs/create",
        {
            "experiment_id": MLFLOW_EXPERIMENT_ID,
            "run_name": "customer-churn-training",
            "start_time": int(datetime.now(timezone.utc).timestamp() * 1000),
        },
    )

    run_id = run["run"]["info"]["run_id"]
    artifact_uri = run["run"]["info"]["artifact_uri"]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")

        # Save the trained model.
        joblib.dump(model, model_path)

        s3 = boto3.client("s3", region_name="us-east-1")

        # Preserve our existing S3 model upload.
        s3.upload_file(model_path, BUCKET, model_key)

        # Upload the same model to the MLflow run.
        upload_mlflow_artifact(artifact_uri, model_path, "model")

        # Prepare an MLflow-compatible model directory.
        package_dir = os.path.join(tmpdir, "mlflow_model")
        os.makedirs(package_dir)

        with open(os.path.join(package_dir, "model.pkl"), "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mlmodel = f"""artifact_path: model
flavors:
  python_function:
    loader_module: mlflow.sklearn
    model_path: model.pkl
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
  sklearn:
    pickled_model: model.pkl
    serialization_format: pickle
    sklearn_version: '{sklearn_version}'
mlflow_version: '3.5.1'
model_uuid: '{safe_run_id}'
"""

        with open(os.path.join(package_dir, "MLmodel"), "w") as handle:
            handle.write(mlmodel)

        with open(os.path.join(package_dir, "requirements.txt"), "w") as handle:
            handle.write(
                f"mlflow==3.5.1\n"
                f"scikit-learn=={sklearn_version}\n"
                "pandas\n"
            )

        with open(os.path.join(package_dir, "conda.yaml"), "w") as handle:
            handle.write(
                "name: customer-churn\n"
                "channels:\n"
                "  - conda-forge\n"
                "dependencies:\n"
                f"  - python={sys.version_info.major}.{sys.version_info.minor}\n"
                "  - pip\n"
                "  - pip:\n"
                "      - -r requirements.txt\n"
            )

        with open(os.path.join(package_dir, "python_env.yaml"), "w") as handle:
            handle.write(
                f"python: {sys.version_info.major}."
                f"{sys.version_info.minor}.{sys.version_info.micro}\n"
                "build_dependencies:\n"
                "  - pip\n"
                "  - setuptools\n"
                "  - wheel\n"
                "dependencies:\n"
                "  - -r requirements.txt\n"
            )

        package_key = f"models/customer-churn/{safe_run_id}/mlflow_model"

        # Upload all five package files to S3 AND MLflow.
        filenames = (
            "MLmodel",
            "model.pkl",
            "requirements.txt",
            "conda.yaml",
            "python_env.yaml",
        )

        for filename in filenames:
            package_file = os.path.join(package_dir, filename)

            s3.upload_file(
                package_file,
                BUCKET,
                f"{package_key}/{filename}",
            )

            upload_mlflow_artifact(
                artifact_uri,
                package_file,
                "model",
            )

    model_s3_uri = f"s3://{BUCKET}/{model_key}"
    logging.info("Model saved: %s", model_s3_uri)

    # Log training parameters to MLflow.
    for key, value in {
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "random_state": 42,
    }.items():
        mlflow_api_request(
            "runs/log-parameter",
            {
                "run_id": run_id,
                "key": key,
                "value": str(value),
            },
        )

    # Log useful run metadata.
    for key, value in {
        "model_s3_uri": model_s3_uri,
        "sklearn_version": sklearn_version,
        "airflow_run_id": context["run_id"],
    }.items():
        mlflow_api_request(
            "runs/set-tag",
            {
                "run_id": run_id,
                "key": key,
                "value": str(value),
            },
        )

    predictions = model.predict(data["X_test"]).tolist()
    logging.info("Training completed; %d predictions", len(predictions))

    return {
        "predictions": predictions,
        "mlflow_run_id": run_id,
        "model_s3_uri": model_s3_uri,
        "model_key": model_key,
        "mlflow_package_uri": f"s3://{BUCKET}/{package_key}",
    }


def evaluate_model(**context):
    """Calculate and record classification performance metrics."""
    ti = context["ti"]
    data = ti.xcom_pull(task_ids="preprocess_data")
    training = ti.xcom_pull(task_ids="train_model")

    actual = data["y_test"]
    predicted = training["predictions"]

    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }

    for key, value in metrics.items():
        mlflow_api_request(
            "runs/log-metric",
            {
                "run_id": training["mlflow_run_id"],
                "key": key,
                "value": value,
                "timestamp": int(
                    datetime.now(timezone.utc).timestamp() * 1000
                ),
                "step": 0,
            },
        )

    mlflow_api_request(
        "runs/update",
        {
            "run_id": training["mlflow_run_id"],
            "status": "FINISHED",
            "end_time": int(
                datetime.now(timezone.utc).timestamp() * 1000
            ),
        },
    )

    logging.info("Metrics: %s", metrics)
    return metrics


def quality_gate(**context):
    """Check that metric values are valid; not a production approval."""
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model")

    if not metrics:
        raise AirflowException("No evaluation metrics available")

    for name, value in metrics.items():
        if not 0 <= value <= 1:
            raise AirflowException(f"Invalid metric {name}: {value}")

    logging.info("Quality gate passed; model is NOT production-approved")


def log_and_register_model(**context):
    """Register a new version using the existing S3 model package."""
    training = context["ti"].xcom_pull(task_ids="train_model")
    model_source = training["mlflow_package_uri"]

    # Create the model name if it does not exist.
    try:
        mlflow_api_request(
            "registered-models/create",
            {
                "name": REGISTERED_MODEL_NAME,
                "description": "Customer churn classification learning project",
            },
        )
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8")

        if exc.code == 400 and "RESOURCE_ALREADY_EXISTS" in error_body:
            logging.info(
                "Registered model %s already exists",
                REGISTERED_MODEL_NAME,
            )
        else:
            raise AirflowException(
                f"MLflow registered-model creation failed: "
                f"{exc.code}: {error_body}"
            ) from exc

    # Register another version only after the quality gate passes.
    result = mlflow_api_request(
        "model-versions/create",
        {
            "name": REGISTERED_MODEL_NAME,
            "source": model_source,
            "run_id": training["mlflow_run_id"],
            "description": (
                "Passed the demonstration quality gate; "
                "not production-approved"
            ),
            "tags": [
                {
                    "key": "airflow_run_id",
                    "value": context["run_id"],
                },
                {
                    "key": "model_s3_uri",
                    "value": training["model_s3_uri"],
                },
            ],
        },
    )

    version = result["model_version"]["version"]

    logging.info(
        "Registered %s version %s from %s",
        REGISTERED_MODEL_NAME,
        version,
        model_source,
    )

    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "version": version,
        "source": model_source,
    }


# Preserve the existing seven-task Airflow pipeline.
with DAG(
    dag_id="customer_churn_pipeline",
    start_date=datetime(2026, 10, 7),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "customer-churn"],
) as dag:

    load = PythonOperator(
        task_id="load_customer_data",
        python_callable=load_customer_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    gate = PythonOperator(
        task_id="quality_gate",
        python_callable=quality_gate,
    )

    register = PythonOperator(
        task_id="log_and_register_model",
        python_callable=log_and_register_model,
    )

    load >> validate >> preprocess >> train >> evaluate >> gate >> register
