# mlops-production-lab
# Customer Churn — End-to-End MLOps Project

## 1. Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for predicting customer churn.

It combines AWS S3, EventBridge, SQS, Apache Airflow, Kubernetes, DVC, Python, scikit-learn, MLflow, GitHub, and FastAPI.

The goal is to automate data ingestion, dataset versioning, validation, preprocessing, model training, evaluation, model registration, and prediction serving.

This is a learning and demonstration MVP, not a production-hardened platform.

## 2. Architecture

Data ingestion and orchestration:

AWS S3 (customer dataset)
    |
    v
Amazon EventBridge (object-created event)
    |
    v
Amazon SQS (event queue)
    |
    v
Airflow SQS Listener DAG
    |
    v
Airflow Customer Churn Pipeline
    |
    +--> Load customer data
    +--> Version dataset with DVC
    +--> Validate data
    +--> Preprocess data
    +--> Train model
    +--> Evaluate model
    +--> Quality gate
    +--> Log and register model
                         |
                         v
                   MLflow Registry
                         |
                         v
                  FastAPI Prediction API

Infrastructure:
- AWS EC2 hosts a local kind Kubernetes cluster.
- Kubernetes hosts Apache Airflow and MLflow.
- Amazon S3 stores input datasets, DVC objects/metadata, and model artifacts.
- GitHub stores the application and pipeline source code.

## 3. Technologies and Responsibilities

| Technology | Responsibility |
| --- | --- |
| Git and GitHub | Version control for source code, DAGs, scripts, and tracked DVC pointer files |
| Amazon S3 | Stores input data, DVC objects, DVC metadata, and MLflow model artifacts |
| EventBridge | Detects S3 object-created events and forwards matching events |
| Amazon SQS | Queues events for the Airflow listener |
| Apache Airflow | Orchestrates and monitors the ML pipeline |
| Kubernetes (kind) | Runs containerized Airflow and MLflow components |
| DVC | Versions datasets and manages data objects in remote storage |
| Python and scikit-learn | Data processing and machine learning |
| MLflow | Tracks experiments, metrics, artifacts, and model versions |
| FastAPI | Exposes an HTTP endpoint for model predictions |

## 4. Airflow Pipeline — Eight Tasks

1. load_customer_data — Reads the customer dataset from S3.
2. version_dataset_with_dvc — Versions the dataset and uploads DVC objects and metadata to S3.
3. validate_data — Checks the input data before processing.
4. preprocess_data — Prepares features and labels for model training.
5. train_model — Trains a customer churn classification model.
6. evaluate_model — Calculates evaluation metrics.
7. quality_gate — Applies demonstration checks before registration.
8. log_and_register_model — Logs results and registers the model in MLflow.

The main DAG is defined in dags/customer_churn_pipeline.py.

The listener DAG, sqs_customer_churn_listener, receives S3 events from SQS and triggers the main pipeline.

## 5. Understanding DVC

Git is designed primarily for versioning source code. DVC helps version data without storing large dataset contents directly in Git.

A typical DVC workflow is:

1. Create or update a dataset.
2. Run dvc add to track its contents.
3. Commit the generated .dvc pointer file to Git.
4. Run dvc push to upload dataset contents to remote storage such as S3.
5. Use Git and DVC together to recover a known code-and-data version.

Example commands for learning:

    dvc add data/customer_data.csv
    git add data/customer_data.csv.dvc
    git commit -m "Track customer dataset with DVC"
    dvc push

These commands illustrate the conventional workflow. They are not required to run the automated pipeline.

In this project, scripts/version_dataset.py performs automated dataset versioning inside the Airflow workflow. It downloads the dataset, calculates its SHA-256 fingerprint, adds it to DVC, pushes the data to S3, and stores the run-specific DVC pointer and JSON manifest under the S3 dvc-metadata/ prefix.

Important distinction: the automatically generated per-run DVC metadata is stored in S3. It is not automatically committed to GitHub.

## 6. Dataset Storage and Traceability

S3 bucket:

    mlops-dvc-sunil-20261007

Input dataset:

    datasets/customer_data.csv

DVC objects:

    files/md5/

Per-run metadata:

    dvc-metadata/

The registered model contains tags recording the source dataset URI, SHA-256 fingerprint, and location of its DVC metadata.

This helps trace a registered model back to its dataset snapshot.

## 7. MLflow Model Registration

The successful demonstration registered:

- Registered model: CustomerChurnModel
- Model version: 6
- Source dataset: s3://mlops-dvc-sunil-20261007/datasets/customer_data.csv
- Dataset SHA-256: 14575bd9804ff8f2dde0337e6eb13625bc6f6c883e4df1e3ba02c99fc65ec4f4

The registered version also contains a dvc_metadata_s3 tag pointing to its DVC pointer file in S3.

The model description explicitly states that it passed the demonstration quality gate and is not production-approved.

## 8. FastAPI Prediction Service

The API source is located at:

    api/app.py

FastAPI provides an HTTP interface for requesting customer churn predictions.

A successful POST /predict request was tested during the project.

The API source code is preserved in GitHub. The local Python virtual environment is intentionally not committed.

## 9. Demonstrated Result

On October 23, 2026, all eight tasks of the customer churn Airflow pipeline completed successfully in a verified DAG run.

The same verification confirmed:
- DVC pointer and JSON manifest objects existed in S3.
- CustomerChurnModel version 6 was visible in MLflow.
- The model version contained dataset URI, SHA-256, and DVC metadata tags.
- Pipeline source code, DVC script, synthetic sample dataset, and FastAPI source were preserved in GitHub.

## 10. Repository Structure

    mlops-production-lab/
    |-- dags/
    |-- scripts/
    |-- data/
    |-- api/
    |   |-- app.py
    |-- Dockerfile.dvc
    |-- customer_data_v2.csv
    |-- README.md

Other files may also be present in the repository.

## 11. MVP Limitations

This is an educational implementation.

- The quality gate checks basic metric validity rather than enforcing business-defined performance thresholds.
- The model registration code retrieves the dataset fingerprint from S3 at registration time. For strict training-data provenance, the exact input fingerprint should instead be carried through from ingestion to registration.
- Infrastructure is not configured for production-grade high availability, security hardening, disaster recovery, or continuous operation.
- A local kind cluster on EC2 was used for learning; it is not an EKS production deployment.
- Terminating the EC2 instance can remove local Kubernetes, Airflow, and MLflow database state even when GitHub and S3 artifacts remain available.

## 12. Learning Outcomes

This project demonstrates how an MLOps engineer can:
- Automate an ML workflow using Airflow.
- Integrate S3 events, EventBridge, and SQS.
- Track and store dataset versions using DVC.
- Train and evaluate a classification model.
- Record model artifacts and versions using MLflow.
- Build a basic prediction API with FastAPI.
- Operate containerized MLOps components on Kubernetes.
- Use GitHub to preserve reproducible source code.

## 13. Cost and Cleanup

The EC2 instance is intended as temporary lab infrastructure.

Before terminating it, verify that source code, essential configuration, required datasets, model artifacts, and any desired database state are backed up.

After termination, review EBS volumes, snapshots, Elastic IPs, load balancers, S3, SQS, and other AWS resources separately, because EC2 termination does not automatically remove every billable resource.

## 14. Project Status

Customer Churn MLOps MVP: Successfully demonstrated.

Next planned learning project: AIOps Anomaly Detection.

