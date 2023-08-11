import os

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

import streamlit as st
import pandas as pd
import shap
from streamlit_shap import st_shap
import numpy as np
import json
import joblib

# from predict import make_prediction, preprocess_data

# sys.path.append(os.path.abspath(os.path.join("..", "model")))

# from main.model.model_loader import load_model


def load_models(model_dir):
    # Load all models from the model directory
    models = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".joblib"):
            model_name = os.path.splitext(filename)[0]
            model_path = os.path.join(model_dir, filename)
            try:
                model = joblib.load(model_path)
                models[model_name] = model
            except Exception as e:
                raise RuntimeError(f"Error loading model: {model_name} - {str(e)}")

    if not models:
        raise RuntimeError("Models not found")

    return models


def load_model(model_path):
    model = joblib.load(model_path)
    return model


models = load_models(os.path.join("..", "model"))

statistics_path = os.path.join("statistics.json")


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


def generate_shap_plot(model_name, data):
    model_path = os.path.join("..", "model", f"{model_name}.joblib")
    # Load the selected model from the selected_model
    model = load_model(model_path)
    # Generate the SHAP plot for the selected model
    explainer = shap.TreeExplainer(model)

    standardize_data = preprocess_data(data, statistics_path)
    shap_values = explainer(standardize_data)

    # Show the SHAP plot
    st_shap(shap.plots.waterfall(shap_values[0], max_display=20), height=600)


def main():
    st.title("Anchor Installation Time Predict")
    st.header("Input Parameters")
    # major_diameter = st.number_input(
    #     "Major Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
    # minor_diameter = st.number_input(
    #     "Minor Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
    undercut = st.number_input(
        "Undercut (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    gap = st.number_input(
        "Gap (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    thread_pitch = st.number_input(
        "Thread Pitch (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    included_angle = st.number_input(
        "Included Angle", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    top_angle = st.number_input(
        "Top Angle", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    # hole_diameter = st.number_input(
    #     "Hole Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
    serration_threads = st.radio(
        "Have Serration Threads", options=[False, True], index=False
    )
    serration_threads = int(serration_threads)  # Convert to int
    length_B = st.number_input(
        "Total length (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    length_C = st.number_input(
        "Thread length (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    embedment_depth = st.number_input(
        "Embedment Depth (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    tip_taper = st.number_input(
        "Tip Taper", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    # shank_diameter = st.number_input(
    #     "Shank Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    # )
    torque_tool = st.selectbox("Torque Tool", options=["Low", "Medium", "High"])
    torque_values = {"Low": 225, "Medium": 300, "High": 500}
    torque_tool = torque_values.get(torque_tool, 225)  # Default to Low torque
    concrete = st.number_input("Concrete Strength (psi)", min_value=0.0, value=0.0)

    input_data = {
        # "Major Diameter": major_diameter,
        # "Minor Diameter": minor_diameter,
        "Pitch": thread_pitch,
        "Included Angle": included_angle,
        "Top Angle": top_angle,
        # "Hole Diameter": hole_diameter,
        "Serrations": serration_threads,
        "Length (C)": length_C,
        "Length (B)": length_B,
        "Embedment depth": embedment_depth,
        "Tip Taper": tip_taper,
        # "Shank Diameter": shank_diameter,
        "Torque Tool": torque_tool,
        "Concrete": concrete,
        "Undercut": undercut,
        "Gap": gap,
    }

    # Hard code the input data for testing
    # Hilti KH-EZ - Batch 1 - Texas concrete - Low torque

    # input_data = {
    #     "Pitch": 0.52,
    #     "Included Angle": 55,
    #     "Top Angle": 27,
    #     "Serrations": 0,
    #     "Length (C)": 5.5,
    #     "Length (B)": 6.5,
    #     "Embedment depth": 5.5,
    #     "Tip Taper": 0.0,
    #     "Torque Tool": 225,
    #     "Concrete": 5500,
    #     "Undercut": 0.074,
    #     "Gap": 0.074,
    # }

    # THDM - Batch 2 - Texas concrete - High torque

    # input_data = {
    #     "Pitch": 0.278,
    #     "Included Angle": 40,
    #     "Top Angle": 20,
    #     "Serrations": 1,
    #     "Length (C)": 3.5,
    #     "Length (B)": 4,
    #     "Embedment depth": 3.25,
    #     "Tip Taper": 0.275,
    #     "Torque Tool": 500,
    #     "Concrete": 5965,
    #     "Undercut": 0.074,
    #     "Gap": 0.074,
    # }

    # Check if any input value is zero or empty
    if (
        # major_diameter == 0.0
        # or minor_diameter == 0.0
        undercut == 0.0
        or gap == 0.0
        or thread_pitch == 0.0
        or included_angle == 0.0
        or top_angle == 0.0
        # or hole_diameter == 0.0
        or length_B == 0.0
        or length_C == 0.0
        or embedment_depth == 0.0
        # or tip_taper == 0.0
        # or shank_diameter == 0.0
        or concrete == 0.0
    ):
        st.error("Error: Please fill in all the input fields.")
        return

    data_frame = pd.DataFrame([input_data])
    # st.write(data_frame)

    show_results = st.button("Predict")

    # Initialize session state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    if show_results or st.session_state.selected_model is not None:
        st.session_state.selected_model = True
        predictions = make_prediction(data_frame)

        if predictions is not None:
            st.header("Predictions:")

            # Create a table to display the predictions
            table_data = []
            for model_name, prediction in predictions.items():
                table_data.append([model_name, prediction])

            table_df = pd.DataFrame(
                table_data, columns=["Model", "Installing time prediction"]
            )

            # Display the table
            st.table(table_df)

            # Create a radio button for users to select a model
            model_options = list(predictions.keys())
            selected_model = st.radio(
                "Select a model to generate SHAP plot:", model_options
            )

            generate_shap_button_clicked = st.button(
                f"Generate SHAP plot for {selected_model}"
            )

            if generate_shap_button_clicked:
                if selected_model is not None:
                    generate_shap_plot(selected_model, data_frame)


if __name__ == "__main__":
    main()
