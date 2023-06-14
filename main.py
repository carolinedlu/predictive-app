from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Specify the directory where the models are stored
model_directory = "model"

# Load all models from the model directory
models = {}
for filename in os.listdir(model_directory):
    if filename.endswith(".joblib"):
        model_name = os.path.splitext(filename)[0]
        model_path = os.path.join(model_directory, filename)
        model = joblib.load(model_path)
        models[model_name] = model


class Data(BaseModel):
    data: list


@app.post("/api/predict/test")
def predict(data: Data):
    # Reshape the input data
    X = np.array(data.data).reshape(1, -1)

    # Predict and store the results for each model
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(X)
        prediction = np.expm1(prediction)
        predictions[model_name] = prediction.tolist()

    # Format the prediction response as JSON
    response = {"predictions": predictions}
    return response
