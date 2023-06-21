import requests
import streamlit as st
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import RobustScaler
import numpy as np


class Data(BaseModel):
    data: list


def make_prediction(api_endpoint, data_frame):
    # Load the scaler
    numeric_features = list(
        data_frame.select_dtypes(include=[np.number]).columns.values
    )
    scaler = RobustScaler()

    for col in numeric_features:
        data_frame[[col]] = scaler.fit_transform(data_frame[[col]])

    # Convert DataFrame to dummy variables
    data = pd.get_dummies(data_frame)

    # Prepare the request data
    request_data = {
        "data": data.values.tolist()
    }  # Convert the DataFrame to a nested list

    try:
        # Send a POST request to the API endpoint
        response = requests.post(api_endpoint, json=request_data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Parse the response as JSON
        predictions = response.json()["predictions"]
        return predictions
    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred: {e}")
        return None


def main():
    st.title("Anchor Installation Time Predict")
    st.header("Input Parameters")
    major_diameter = st.number_input(
        "Major Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    minor_diameter = st.number_input(
        "Minor Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
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
    hole_diameter = st.number_input(
        "Hole Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
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
    shank_diameter = st.number_input(
        "Shank Diameter (in.)", min_value=0.0, value=0.0, format="%.4f", step=1e-4
    )
    torque_tool = st.selectbox("Torque Tool", options=["Low", "Medium", "High"])
    torque_values = {"Low": 225, "Medium": 300, "High": 500}
    torque_tool = torque_values.get(torque_tool, 225)  # Default to Low torque
    # concrete = st.radio("Concrete", options=["Texas", "West Chicago"])
    # concrete_values = {"Texas": (1, 0), "West Chicago": (0, 1)}
    # concrete_tx, concrete_wc = concrete_values.get(concrete, (1, 0))  # Default to Texas

    input_data = {
        "major_diameter": major_diameter,
        "minor_diameter": minor_diameter,
        "thread_pitch": thread_pitch,
        "included_angle": included_angle,
        "top_angle": top_angle,
        "hole_diameter": hole_diameter,
        "serration_threads": serration_threads,
        "length_B": length_B,
        "length_C": length_C,
        "embedment_depth": embedment_depth,
        "tip_taper": tip_taper,
        "shank_diameter": shank_diameter,
        "torque_tool": torque_tool,
        # "concrete_tx": concrete_tx,
        # "concrete_wc": concrete_wc,
    }

    data_frame = pd.DataFrame([input_data])

    if st.button("Predict"):
        api_endpoint = (
            "http://localhost:5001/api/predict/test"  # Modify the URL if needed
        )
        predictions = make_prediction(api_endpoint, data_frame)
        st.header("Predictions:")

        # Create a table to display the predictions
        table_data = [
            [model_name, prediction] for model_name, prediction in predictions.items()
        ]
        table_df = pd.DataFrame(
            table_data, columns=["Model", "Installing time prediction"]
        )
        st.table(table_df)


if __name__ == "__main__":
    main()
