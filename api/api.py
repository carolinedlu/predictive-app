import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "app")))
from predict import make_prediction, preprocess_data

app = FastAPI()


class Data(BaseModel):
    data: list


@app.post("/api/predict/test")
def predict(data: Data):
    # Reshape the input data if needed
    X = np.array(data.data).reshape(1, -1)

    # Make prediction
    predictions = make_prediction(X)

    # Return the prediction
    return predictions
