import requests
import streamlit as st
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import RobustScaler
import numpy as np


class Data(BaseModel):
    data: list


def make_prediction(data_frame: pd.DataFrame):
    # Define the API endpoint for the FastAPI backend
    api_endpoint = "http://localhost:5001/api/predict/test"  # Modify the URL if needed

    # Load the scaler
    numeric_features = list(
        data_frame.select_dtypes(include=[np.number]).columns.values
    )

    scaler = RobustScaler()

    for col in numeric_features:
        data_frame[[col]] = scaler.fit_transform(data_frame[[col]])

    # Convert dataFrame to dummy variables
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
        "Major Diameter (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    minor_diameter = st.number_input(
        "Minor Diameter (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    thread_pitch = st.number_input(
        "Thread Pitch (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    included_angle = st.number_input(
        "Included Angle", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    top_angle = st.number_input(
        "Top Angle", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    hole_diameter = st.number_input(
        "Hole Diameter (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    serration_threads = st.radio(
        "Have Serration Threads", options=[False, True], index=False
    )
    if serration_threads:
        serration_threads = 1
    else:
        serration_threads = 0
    length_B = st.number_input(
        "Total length (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    length_C = st.number_input(
        "Thread length (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    embedment_depth = st.number_input(
        "Embedment Depth (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    tip_taper = st.number_input(
        "Tip Taper", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    shank_diameter = st.number_input(
        "Shank Diameter (in.)", min_value=0.0, value=0.0, format="%.5f", step=0.0001
    )
    torque_tool = st.selectbox("Torque Tool", options=["Low", "Medium", "High"])
    if torque_tool == "Low":
        torque_tool = 225
    elif torque_tool == "Medium":
        torque_tool = 300
    else:
        torque_tool = 500
    concrete = st.radio("Concrete", options=["Texas", "West Chicago"])
    if concrete == "Texas":
        concrete_tx = 1
        concrete_wc = 0
    else:
        concrete_tx = 0
        concrete_wc = 1

    input = {
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
        "concrete_tx": concrete_tx,
        "concrete_wc": concrete_wc,
    }

    # Input conversion to DataFrame
    data_frame = pd.DataFrame([input])

    if st.button("Predict"):
        predictions = make_prediction(data_frame)
        st.header("Predictions:")

        # Create a table to display the predictions
        table_data = []
        for model_name, prediction in predictions.items():
            table_data.append([model_name, prediction])

        table_df = pd.DataFrame(table_data, columns=["Model", "Prediction"])
        st.table(table_df)


if __name__ == "__main__":
    main()
