from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from error_handler import (
    BadRequestError,
    handle_bad_request_error,
    NotFoundError,
    handle_not_found_error,
)

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

if not models:
    raise NotFoundError("Not found any models")


class Data(BaseModel):
    data: list


@app.exception_handler(BadRequestError)
async def handle_bad_request(request, exc):
    return await handle_bad_request_error(request, exc)


@app.exception_handler(NotFoundError)
async def handle_not_found(request, exc):
    return await handle_not_found_error(request, exc)


@app.post("/api/predict/test")
def predict(data: Data):
    try:
        # Reshape the input data
        X = np.array(data.data).reshape(1, -1)

        # Predict and store the results for each model
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(X)
            prediction = np.expm1(prediction)
            predictions[model_name] = f"{prediction[0]:.6f} seconds"

        # Format the prediction response as JSON
        response = {"predictions": predictions}
        return response
    except Exception as e:
        raise BadRequestError("Invalid request data.")
