import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("winequality-red.csv")  # Adjust this path as per your project
    return data

# Function to train the model
def train_model(data):
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# UI layout
st.title("Red Wine Quality Prediction")
st.write("Upload the dataset or use the default wine quality dataset for prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = load_data()

st.write("Dataset Preview:")
st.write(data.head())

# Train model button
if st.button("Train Model"):
    model, mse, r2 = train_model(data)
    st.write(f"Model trained successfully! Mean Squared Error: {mse}, R2 Score: {r2}")

# User input for prediction
st.write("Make a Prediction")
input_data = []
for column in data.columns[:-1]:  # Skip 'quality' column
    value = st.number_input(f"Enter {column}", min_value=float(data[column].min()), max_value=float(data[column].max()))
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    st.write(f"Predicted Quality: {prediction[0]}")

