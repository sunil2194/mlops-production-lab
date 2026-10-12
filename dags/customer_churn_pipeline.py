
from datetime import datetime
from io import StringIO

import boto3
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


def load_customer_data():
    # Connect to AWS S3
    s3 = boto3.client("s3", region_name="us-east-1")

    # Retrieve the CSV file from S3
    response = s3.get_object(
        Bucket="mlops-dvc-sunil-20261007",
        Key="datasets/customer_data.csv",
    )

    # Read CSV into a Pandas DataFrame
    csv_content = response["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(csv_content))

    print("Successfully loaded customer data from S3!")
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head().to_string(index=False))


with DAG(
    dag_id="customer_churn_pipeline",
    start_date=datetime(2026, 10, 7),
    schedule=None,
    catchup=False,
    tags=["mlops", "customer-churn"],
) as dag:

    load_data = PythonOperator(
        task_id="load_customer_data",
        python_callable=load_customer_data,
    )
