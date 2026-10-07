from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def hello():
    print("Hello from our GitHub-managed Airflow DAG!")


with DAG(
    dag_id="hello_mlops",
    start_date=datetime(2026, 10, 7),
    schedule=None,
    catchup=False,
) as dag:

    hello_task = PythonOperator(
        task_id="hello",
        python_callable=hello,
    )