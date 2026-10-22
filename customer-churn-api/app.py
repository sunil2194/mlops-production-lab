
# ============================================================
# CUSTOMER CHURN PREDICTION API
#
# Purpose:
# Accept customer details through an HTTP API and use our
# already-trained ML model to predict customer churn.
#
# Training happens in Airflow.
# MLflow stores and manages the trained model.
# FastAPI serves predictions using the loaded model.
# ============================================================


# STEP 1: Import required libraries

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.tracking import MlflowClient


# STEP 2: Create the FastAPI application
#
# Uvicorn runs this application and listens for HTTP requests.

app = FastAPI(title="Customer Churn Prediction API")


# STEP 3: Connect to our existing MLflow tracking server
#
# Port 5000 is forwarded from the MLflow Kubernetes service
# to localhost on our EC2 instance.
#
# This does NOT start another MLflow server.

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# STEP 4: Find the latest registered model version
#
# MlflowClient communicates with the MLflow Model Registry.
#
# search_model_versions() returns the registered versions
# of CustomerChurnModel.
#
# max() selects the highest numeric version.
# This is not necessarily the best-performing model.

client = MlflowClient()

versions = client.search_model_versions(
    "name='CustomerChurnModel'"
)

latest = max(versions, key=lambda v: int(v.version))


# STEP 5: Load the trained model into Python memory
#
# latest.run_id identifies the MLflow training run associated
# with the selected registered model version.
#
# runs:/<run_id>/model points to the saved model artifacts.
#
# MLflow reads the model metadata and loads the serialized
# scikit-learn model.
#
# IMPORTANT:
# This code executes when the application starts.
# We do NOT load the model again for every prediction.

model = mlflow.sklearn.load_model(
    f"runs:/{latest.run_id}/model"
)


# STEP 6: Define the expected customer input
#
# Pydantic checks that incoming requests contain:
# age     -> integer
# income  -> number
# balance -> number
#
# Example:
# {"age": 35, "income": 60000, "balance": 12000}

class Customer(BaseModel):
    age: int
    income: float
    balance: float


# STEP 7: Health / welcome endpoint
#
# GET http://127.0.0.1:8000/
#
# This endpoint confirms that FastAPI is responding.

@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}


# STEP 8: Prediction endpoint
#
# POST http://127.0.0.1:8000/predict
#
# This function executes whenever a client sends
# customer details to the /predict endpoint.

@app.post("/predict")
def predict(customer: Customer):

    # Convert the validated customer input into a dictionary,
    # then create a Pandas DataFrame containing one customer.
    #
    # Columns: age, income, balance

    data = pd.DataFrame([customer.model_dump()])

    # Ask the trained model to predict the customer class.
    #
    # 0 = Customer will likely stay
    # 1 = Customer will likely leave
    #
    # predict(data) returns an array of predictions.
    # [0] retrieves the first customer's prediction.
    # int() converts it into a normal Python integer.

    prediction = int(model.predict(data)[0])

    # Calculate the estimated probability of churn.
    #
    # predict_proba(data) returns probabilities for each class.
    #
    # Example: [[0.8285, 0.1715]]
    #
    # First [0]  -> First customer's row
    # Second [1] -> Probability column for class 1 (churn)
    #
    # Our training labels were:
    # No -> 0 (stay)
    # Yes -> 1 (churn)
    #
    # float() converts the result into a Python float.

    probability = float(model.predict_proba(data)[0][1])

    # Return the prediction and churn probability as JSON.
    #
    # Example:
    # {
    #     "churn_prediction": 0,
    #     "churn_probability": 0.1715
    # }

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4)
    }
