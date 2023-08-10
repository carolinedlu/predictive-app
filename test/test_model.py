import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "app")))
sys.path.append(os.path.abspath(os.path.join("..", "model")))
import numpy as np
from predict import preprocess_data
from model_loader import load_models
import pandas as pd

statistics_path = os.path.join("..", "app", "statistics.json")


def load_data(excel_file_path, num_features):
    try:
        data = pd.read_excel(excel_file_path, sheet_name=f"feature_{num_features}")
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from Excel: {str(e)}")


def model_number_features(model):
    return model.n_features_in_


def main():
    model_dir = os.path.join("..", "model")
    models = load_models(model_dir)

    xlsx_file_path = os.path.join(os.path.dirname(__file__), "test_data.xlsx")

    for model_name, chosen_model in models.items():
        num_features = model_number_features(chosen_model)
        data = load_data(xlsx_file_path, num_features)
        standardized_data = preprocess_data(data.copy(), statistics_path)
        sample_data = standardized_data.to_numpy()

        try:
            prediction = chosen_model.predict(sample_data)
            prediction = np.expm1(prediction)
            print(f"Model: {model_name}, Prediction: {prediction}")
        except Exception as e:
            print(f"Error occurred during prediction for {model_name}: {e}")


if __name__ == "__main__":
    main()
