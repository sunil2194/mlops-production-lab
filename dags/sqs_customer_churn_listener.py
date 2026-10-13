
import json
import logging
import time
from datetime import datetime, timedelta

import boto3
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.api.common.trigger_dag import trigger_dag
from airflow.exceptions import DagRunAlreadyExists
from airflow.models.dagrun import DagRun
from airflow.utils.session import create_session
from airflow.utils.state import DagRunState

QUEUE_URL = (
    "https://sqs.us-east-1.amazonaws.com/"
    "000515319532/mlops-customer-data-events"
)
REGION = "us-east-1"
TARGET_DAG = "customer_churn_pipeline"

VISIBILITY_SECONDS = 300
POLL_SECONDS = 10
MAX_WAIT_SECONDS = 3600


def get_run_state(run_id):
    with create_session() as session:
        run = (
            session.query(DagRun)
            .filter(
                DagRun.dag_id == TARGET_DAG,
                DagRun.run_id == run_id,
            )
            .first()
        )
        return run.state if run else None


def process_sqs_event():
    sqs = boto3.client("sqs", region_name=REGION)

    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=10,
        VisibilityTimeout=VISIBILITY_SECONDS,
    )

    messages = response.get("Messages", [])

    if not messages:
        raise AirflowSkipException("No SQS messages available")

    message = messages[0]
    receipt_handle = message["ReceiptHandle"]
    event = json.loads(message["Body"])

    if (
        event.get("source") != "aws.s3"
        or event.get("detail-type") != "Object Created"
    ):
        raise ValueError("Unexpected SQS event")

    bucket = event["detail"]["bucket"]["name"]
    key = event["detail"]["object"]["key"]
    event_id = event["id"]

    run_id = f"s3_event__{event_id}"

    logging.info("Processing s3://%s/%s", bucket, key)

    state = get_run_state(run_id)

    if state == DagRunState.SUCCESS:
        logging.info("Event already processed successfully")

    elif state == DagRunState.FAILED:
        raise RuntimeError(
            f"Existing DAG run {run_id} failed; manual recovery required"
        )

    elif state is None:
        try:
            trigger_dag(
                dag_id=TARGET_DAG,
                run_id=run_id,
                conf={
                    "bucket": bucket,
                    "key": key,
                },
            )
            logging.info("Triggered DAG run %s", run_id)
        except DagRunAlreadyExists:
            logging.info("DAG run already exists: %s", run_id)

    start = time.monotonic()

    while True:
        state = get_run_state(run_id)

        if state == DagRunState.SUCCESS:
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=receipt_handle,
            )
            logging.info("DAG succeeded; SQS event acknowledged")
            return

        if state == DagRunState.FAILED:
            raise RuntimeError(
                f"Customer Churn DAG failed: {run_id}"
            )

        if time.monotonic() - start >= MAX_WAIT_SECONDS:
            raise TimeoutError(
                f"Timed out waiting for DAG run {run_id}"
            )

        sqs.change_message_visibility(
            QueueUrl=QUEUE_URL,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=VISIBILITY_SECONDS,
        )

        time.sleep(POLL_SECONDS)


with DAG(
    dag_id="sqs_customer_churn_listener",
    start_date=datetime(2026, 10, 13),
    schedule="* * * * *",
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["mlops", "sqs", "event-driven"],
) as dag:

    process_event = PythonOperator(
        task_id="process_sqs_event",
        python_callable=process_sqs_event,
        execution_timeout=timedelta(minutes=65),
    )
