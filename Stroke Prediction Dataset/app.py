import streamlit as st
import pandas as pd
import joblib


# 1. Load trained model

try:
    model = joblib.load("stroke_prediction_model.pkl")
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")


# 2. Website title

st.title("🧠 Stroke Prediction System")

st.write(
    "Enter the person's information below to get a prediction from the trained machine learning model."
)

if model_loaded:
    st.success("Model loaded successfully!")


# 3. Patient Information Form

st.header("Patient Information")


with st.form("stroke_prediction_form"):

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

    with col2:
        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=120.0,
            value=30.0,
            step=1.0
        )


    # Row 2
    col1, col2 = st.columns(2)

    with col1:
        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

    with col2:
        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"]
        )


    # Row 3
    col1, col2 = st.columns(2)

    with col1:
        ever_married = st.selectbox(
            "Ever Married",
            ["No", "Yes"]
        )

    with col2:
        work_type = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ]
        )


    # Row 4
    col1, col2 = st.columns(2)

    with col1:
        residence_type = st.selectbox(
            "Residence Type",
            ["Urban", "Rural"]
        )

    with col2:
        avg_glucose_level = st.number_input(
            "Average Glucose Level",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=0.1
        )


    # Row 5
    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

    with col2:
        smoking_status = st.selectbox(
            "Smoking Status",
            [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ]
        )



    # Prediction Button


    submit = st.form_submit_button(
        "🔍 Predict Stroke"
    )


# 4. Prediction

if submit:

    # Convert Yes/No values to 0/1

    hypertension = 1 if hypertension == "Yes" else 0

    heart_disease = 1 if heart_disease == "Yes" else 0

    ever_married = 1 if ever_married == "Yes" else 0


    # Convert Gender

    gender_mapping = {
        "Female": 0,
        "Male": 1,
        "Other": 2
    }

    gender = gender_mapping[gender]


    # Convert Work Type

    work_type_mapping = {
        "Govt_job": 0,
        "Never_worked": 1,
        "Private": 2,
        "Self-employed": 3,
        "children": 4
    }

    work_type = work_type_mapping[work_type]


    # Convert Residence Type

    residence_mapping = {
        "Rural": 0,
        "Urban": 1
    }

    residence_type = residence_mapping[residence_type]


    # Convert Smoking Status

    smoking_mapping = {
        "Unknown": 0,
        "formerly smoked": 1,
        "never smoked": 2,
        "smokes": 3
    }

    smoking_status = smoking_mapping[smoking_status]



    # 5. Create input DataFrame


    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    })



    # 6. Make prediction


    if model_loaded:

        try:

            prediction = model.predict(input_data)[0]

            # Show result

            st.divider()

            st.header("Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ The model predicts a possible stroke."
                )

                st.warning(
                    "This is a machine-learning prediction, not a medical diagnosis."
                )

            else:

                st.success(
                    "✅ The model predicts no stroke."
                )

                st.info(
                    "This is a machine-learning prediction, not a medical diagnosis."
                )


        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )

            st.write("Input data sent to the model:")
            st.dataframe(input_data)