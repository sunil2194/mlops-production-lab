from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="hello_mlops",
    start_date=datetime(2024, 10, 13),
    schedule=None,
    catchup=False,
)
def hello_mlops():

    @task
    def say_hello():
        print("Hello from my first MLOps DAG!")

    say_hello()


hello_mlops()
