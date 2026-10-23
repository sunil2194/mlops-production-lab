
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

app = FastAPI(title="Customer Churn Prediction API")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Find the latest registered model version
client = MlflowClient()
versions = client.search_model_versions(
    "name='CustomerChurnModel'"
)
latest = max(versions, key=lambda v: int(v.version))

# Load its trained model from MLflow run artifacts
model = mlflow.sklearn.load_model(
    f"runs:/{latest.run_id}/model"
)

class Customer(BaseModel):
    age: int
    income: float
    balance: float

@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}

@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4)
    }
