
import json
import logging
from datetime import datetime, timedelta

import boto3
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import (
    TriggerDagRunOperator,
)

QUEUE_URL = (
   "https://sqs.us-east-1.amazonaws.com/"
    "000515319532/mlops-customer-churn-poc-queue"
)

REGION = "us-east-1"
TARGET_DAG = "customer_churn_pipeline"


def receive_s3_event(**context):
    sqs = boto3.client("sqs", region_name=REGION)

    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=10,
        VisibilityTimeout=43200,
    )

    messages = response.get("Messages", [])

    if not messages:
        raise AirflowSkipException("No SQS messages available")

    message = messages[0]
    event = json.loads(message["Body"])

    if (
        event.get("source") != "aws.s3"
        or event.get("detail-type") != "Object Created"
    ):
        raise ValueError("Unexpected SQS event")

    bucket = event["detail"]["bucket"]["name"]
    key = event["detail"]["object"]["key"]
    event_id = event["id"]

    logging.info(
        "Received S3 event %s for s3://%s/%s",
        event_id,
        bucket,
        key,
    )

    return {
        "bucket": bucket,
        "key": key,
        "event_id": event_id,
        "receipt_handle": message["ReceiptHandle"],
    }


def acknowledge_sqs_message(**context):
    event = context["ti"].xcom_pull(
        task_ids="receive_s3_event"
    )

    if not event:
        raise AirflowSkipException("No event to acknowledge")

    sqs = boto3.client("sqs", region_name=REGION)

    sqs.delete_message(
        QueueUrl=QUEUE_URL,
        ReceiptHandle=event["receipt_handle"],
    )

    logging.info(
        "SQS event %s successfully acknowledged",
        event["event_id"],
    )


with DAG(
    dag_id="sqs_customer_churn_listener",
    start_date=datetime(2026, 10, 13),
    schedule="* * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "sqs", "event-driven"],
) as dag:

    receive_event = PythonOperator(
        task_id="receive_s3_event",
        python_callable=receive_s3_event,
    )

    trigger_pipeline = TriggerDagRunOperator(
        task_id="trigger_customer_churn",
        trigger_dag_id=TARGET_DAG,
        trigger_run_id=(
            "s3_event__"
            "{{ ti.xcom_pull(task_ids='receive_s3_event')['event_id'] }}"
        ),
        conf={
            "bucket": "{{ ti.xcom_pull(task_ids='receive_s3_event')['bucket'] }}",
            "key": "{{ ti.xcom_pull(task_ids='receive_s3_event')['key'] }}",
        },
        wait_for_completion=True,
        poke_interval=10,
        skip_when_already_exists=True,
    )

    acknowledge = PythonOperator(
        task_id="acknowledge_sqs_message",
        python_callable=acknowledge_sqs_message,
    )

    receive_event >> trigger_pipeline >> acknowledge
