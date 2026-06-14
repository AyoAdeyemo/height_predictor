import streamlit as st
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "height_predictor_lr.pkl")

model = joblib.load(model_path)

st.title("Height Prediction App")

# -------------------
# User Inputs
# -------------------

weight = st.number_input("Weight", min_value=30.0, max_value=250.0, value=70.0)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age_cat = st.selectbox(
    "Age Category",
    ["Young Adult", "Adult", "Senior"]
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

# -------------------
# Encode Inputs
# -------------------

gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

smoker_0 = 1 if smoker == "No" else 0
smoker_1 = 1 if smoker == "Yes" else 0

# -------------------
# Create dataframe
# -------------------

input_data = pd.DataFrame({
    "Weight": [weight],
    "Gender_Female": [gender_female],
    "Gender_Male": [gender_male],
    "Smoker_0": [smoker_0],
    "Smoker_1": [smoker_1],
    "Age_Category_Young Adult": [1 if age_cat == "Young Adult" else 0],
    "Age_Category_Adult": [1 if age_cat == "Adult" else 0],
    "Age_Category_Senior": [1 if age_cat == "Senior" else 0]
})

# -------------------
# Predict
# -------------------

if st.button("Predict Height"):

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Height: {prediction:.2f} cm")

    height_m = prediction[0] / 100

    bmi = weight / (height_m ** 2)

    st.metric(
        "Estimated BMI",
        f"{bmi:.1f}"
    )