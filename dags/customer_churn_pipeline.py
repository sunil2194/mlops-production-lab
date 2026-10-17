
from datetime import datetime
from io import StringIO
import logging
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import os
import tempfile
import joblib
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
MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
MLFLOW_EXPERIMENT_ID = "1"
DEFAULT_BUCKET = "mlops-dvc-sunil-20261007"
DEFAULT_KEY = "datasets/customer_data.csv"

FEATURES = ["age", "income", "balance"]
REQUIRED = ["customer_id", *FEATURES, "left_bank"]

def mlflow_api_request(endpoint, payload):
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/{endpoint}"

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))
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

# context is a dictionary that contains information about the current execution context of the Airflow task. It provides access to various runtime parameters and allows tasks to communicate with each other through XComs (cross-communication). In this function, we use the context to retrieve the preprocessed data from the previous task and to log information about the model training process.
def train_model(**context):
    #ti stands for Task Instance,We use ti here to access XCom, which allows Airflow tasks to exchange small pieces of data.
    ti = context["ti"]
    #Retrieve Preproccessed Data from XCom,The preprocess_data task returns a dictionary containing the training and testing datasets. We pull this data from XCom to use it for model training.
    data = ti.xcom_pull(task_ids="preprocess_data")
    #Create the machine learning pipeline,The make_pipeline function from scikit-learn is used to create a machine learning pipeline. This pipeline consists of two steps: StandardScaler for feature scaling and LogisticRegression for classification. The max_iter parameter is set to 1000 to allow sufficient iterations for convergence, and random_state is set for reproducibility.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    #Train the model,The model is trained using the fit method, which takes the training features (X_train) and labels (y_train) as input. This step adjusts the model parameters based on the training data. 
    model.fit(data["X_train"], data["y_train"])
    #make predictions on the test set,After training, the model is used to make predictions on the test features (X_test). The predictions are converted to a list for easier handling and logging.
    
    # Create a unique model file for this Airflow run
   
    mlflow_run_id = mlflow_run["run"]["info"]["run_id"]

    logging.info("MLflow Run ID: %s", mlflow_run_id)

    # Log model parameters to MLflow
    for param_name, param_value in {
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "random_state": 42,
    }.items():
        mlflow_api_request(
            "runs/log-parameter",
            {
                "run_id": mlflow_run_id,
                "key": param_name,
                "value": str(param_value),
            },
        )

    # Record the model's S3 location in MLflow
    mlflow_api_request(
        "runs/set-tag",
        {
            "run_id": mlflow_run_id,
            "key": "model_s3_uri",
            "value": model_s3_uri,
        },
    )

    # Record the scikit-learn version
    mlflow_api_request(
        "runs/set-tag",
        {
            "run_id": mlflow_run_id,
            "key": "sklearn_version",
            "value": "1.8.0",
        },
    )

    # Predict using the trained model
    predictions = model.predict(data["X_test"]).tolist()

    logging.info("Model training completed")
    logging.info("Predictions: %s", predictions)

    return {
        "predictions": predictions,
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "mlflow_run_id": mlflow_run_id,
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

    mlflow_run_id = training["mlflow_run_id"]

    for metric_name, metric_value in metrics.items():
        mlflow_api_request(
            "runs/log-metric",
            {
                "run_id": mlflow_run_id,
                "key": metric_name,
                "value": metric_value,
                "timestamp": int(datetime.now().timestamp() * 1000),
                "step": 0,
            },
        )

    mlflow_api_request(
        "runs/update",
        {
            "run_id": mlflow_run_id,
            "status": "FINISHED",
            "end_time": int(datetime.now().timestamp() * 1000),
        },
    )

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

    mlflow_run_id = training["mlflow_run_id"]

    for metric_name, metric_value in metrics.items():
        mlflow_api_request(
            "runs/log-metric",
            {
                "run_id": mlflow_run_id,
                "key": metric_name,
                "value": metric_value,
                "timestamp": int(datetime.now().timestamp() * 1000),
                "step": 0,
            },
        )

    mlflow_api_request(
        "runs/update",
        {
            "run_id": mlflow_run_id,
            "status": "FINISHED",
            "end_time": int(datetime.now().timestamp() * 1000),
        },
    )

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
