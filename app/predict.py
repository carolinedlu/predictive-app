import numpy as np
import os
import json

from model_loader import load_models


models = load_models(os.path.join("..", "model"))


def preprocess_data(data_frame, statistics_path):
    with open(statistics_path) as f:
        statistics = json.load(f)

    try:
        data = data_frame.copy()

        for column in data.columns:
            # print(data[column].dtype)
            if column in statistics:
                if data[column].dtype == np.float64 or data[column].dtype == np.int64:
                    data[column] = (data[column] - statistics[column]["median"]) / (
                        statistics[column]["q75"] - statistics[column]["q25"]
                    )
            else:
                print(f"Skipping standardization for '{column}' column")

        print(data)

        return data
    except Exception as e:
        raise RuntimeError(f"Error preprocessing data: {str(e)}")


class PredictionError(Exception):
    def __init__(self, model_name, original_exception):
        self.model_name = model_name
        self.original_exception = original_exception
        super().__init__(
            f"Error predicting with model: {model_name} - {str(original_exception)}"
        )


def make_prediction(data_frame):
    data = preprocess_data(data_frame, os.path.join("statistics.json"))
    X = np.array(data).reshape(1, -1)

    predictions = {}

    for model_name, model in models.items():
        try:
            prediction = model.predict(X)
            prediction = np.expm1(prediction)
            predictions[model_name] = f"{prediction[0]:.6f} seconds"
        except (ValueError, TypeError) as e:
            raise PredictionError(model_name, e)

    return predictions
