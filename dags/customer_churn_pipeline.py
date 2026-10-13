
from datetime import datetime
from io import StringIO
import logging

import boto3
import pandas as pd

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DEFAULT_BUCKET = "mlops-dvc-sunil-20261007"
DEFAULT_KEY = "datasets/customer_data.csv"

FEATURES = ["age", "income", "balance"]
REQUIRED = ["customer_id", *FEATURES, "left_bank"]


def load_customer_data(**context):
    conf = context["dag_run"].conf or {}

    bucket = conf.get("bucket", DEFAULT_BUCKET)
    key = conf.get("key", DEFAULT_KEY)

    logging.info("Reading s3://%s/%s", bucket, key)

    s3 = boto3.client("s3", region_name="us-east-1")
    response = s3.get_object(Bucket=bucket, Key=key)

    df = pd.read_csv(
        StringIO(response["Body"].read().decode("utf-8"))
    )

    logging.info("Loaded %s customer records", len(df))

    return df.to_json(orient="split")


def validate_data(**context):
    ti = context["ti"]

    df = pd.read_json(
        StringIO(ti.xcom_pull(task_ids="load_customer_data")),
        orient="split",
    )

    missing = [col for col in REQUIRED if col not in df.columns]

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

    if df["left_bank"].value_counts().min() < 2:
        raise AirflowException(
            "At least two examples per class are required"
        )

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="raise"
        )

    logging.info("Data validation passed")

    return df.to_json(orient="split")


def preprocess_data(**context):
    ti = context["ti"]

    df = pd.read_json(
        StringIO(ti.xcom_pull(task_ids="validate_data")),
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

    logging.info(
        "Training records: %s; Testing records: %s",
        len(X_train),
        len(X_test),
    )

    return {
        "X_train": X_train.values.tolist(),
        "X_test": X_test.values.tolist(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
    }


def train_model(**context):
    ti = context["ti"]
    data = ti.xcom_pull(task_ids="preprocess_data")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )

    model.fit(data["X_train"], data["y_train"])

    predictions = model.predict(data["X_test"]).tolist()

    logging.info("Model training completed")
    logging.info("Predictions: %s", predictions)

    return {
        "predictions": predictions,
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
    }


def evaluate_model(**context):
    ti = context["ti"]

    data = ti.xcom_pull(task_ids="preprocess_data")
    training = ti.xcom_pull(task_ids="train_model")

    actual = data["y_test"]
    predicted = training["predictions"]

    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
    }

    logging.info("Model evaluation: %s", metrics)

    return metrics


def quality_gate(**context):
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_model")

    if not metrics:
        raise AirflowException("No evaluation metrics available")

    for name, value in metrics.items():
        if not 0 <= value <= 1:
            raise AirflowException(
                f"Invalid metric {name}: {value}"
            )

    logging.info("Pipeline execution quality gate passed")
    logging.info(
        "Learning dataset only: model is not production-approved"
    )


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

    load >> validate >> preprocess >> train >> evaluate >> gate
