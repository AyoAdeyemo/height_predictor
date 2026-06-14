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

age = st.number_input("Age", min_value=18, max_value=120, value=30)

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

activity_level = st.selectbox(
    "Activity Level",
    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
)

# -------------------
# Manual Encoding
# -------------------

gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

smoker_0 = 1 if smoker == "No" else 0
smoker_1 = 1 if smoker == "Yes" else 0

# age_young = 1 if age_cat == "Young Adult" else 0
# age_adult = 1 if age_cat == "Adult" else 0
# age_senior = 1 if age_cat == "Senior" else 0

# Activity Level encoding (IMPORTANT FIX)
activity_sedentary = 1 if activity_level == "Sedentary" else 0
activity_light = 1 if activity_level == "Lightly Active" else 0
activity_moderate = 1 if activity_level == "Moderately Active" else 0
activity_very = 1 if activity_level == "Very Active" else 0

# -------------------
# Create DataFrame
# -------------------

input_data = pd.DataFrame({
    "Weight": [weight],

    "Age": [age],

    "Gender_Female": [gender_female],
    "Gender_Male": [gender_male],

    "Smoker_0": [smoker_0],
    "Smoker_1": [smoker_1],

    

    # "Age_Category_Young Adult": [age_young],
    # "Age_Category_Adult": [age_adult],
    # "Age_Category_Senior": [age_senior],

    "Activity_Level_Sedentary": [activity_sedentary],
    "Activity_Level_Lightly Active": [activity_light],
    "Activity_Level_Moderately Active": [activity_moderate],
    "Activity_Level_Very Active": [activity_very],
})

# align with model
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# -------------------
# Predict
# -------------------

if st.button("Predict Height"):

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Height: {prediction:.2f} cm")

    height_m = prediction / 100
    bmi = weight / (height_m ** 2)

    st.metric(
        "Estimated BMI",
        f"{bmi:.1f}"
    )