import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('Big_Mart_Sales_Prediction.pkl')

# Define encoders for categorical features (these should match the encoders used during training)
outlet_identifier_encoder = LabelEncoder()
outlet_identifier_encoder.fit(["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"])

outlet_size_encoder = LabelEncoder()
outlet_size_encoder.fit(["High", "Medium", "Small"])

outlet_type_encoder = LabelEncoder()
outlet_type_encoder.fit(["Grocery Store", "Super Market 1", "Super Market2", "Super Market 3"])

# Streamlit UI
st.title("Big Mart Sales Prediction")
st.write("Predict sales based on input features.")

# User inputs
item_mrp = st.number_input("Item MRP:", min_value=0.0, step=0.1)

outlet_identifier = st.selectbox(
    "Outlet Identifier",
    ["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"]
)

outlet_size = st.selectbox(
    "Outlet Size",
    ["High", "Medium", "Small"]
)

outlet_type = st.selectbox(
    "Outlet Type",
    ["Grocery Store", "Super Market 1", "Super Market2", "Super Market 3"]
)

establishment_year = st.number_input(
    "Establishment Year:",
    min_value=1900,
    max_value=2024,
    step=1
)

# Collect the inputs for prediction
if st.button("Predict"):
    # Encode categorical variables
    encoded_outlet_identifier = outlet_identifier_encoder.transform([outlet_identifier])[0]
    encoded_outlet_size = outlet_size_encoder.transform([outlet_size])[0]
    encoded_outlet_type = outlet_type_encoder.transform([outlet_type])[0]

    # Prepare input data as a numpy array
    input_data = np.array([[item_mrp, encoded_outlet_identifier, encoded_outlet_size, encoded_outlet_type, establishment_year]])

    # Predict using the model
    prediction = model.predict(input_data)

    # Display the prediction result
    st.write(f"Predicted Sales: {prediction[0]}")
