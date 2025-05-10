import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
@st.cache_data
def load_model():
    df = pd.read_csv(r"C:\Users\prash\DataScience\DataScience\MachineLearning\Class2\Assignment2\hiring.csv")
    # Filling the nan values
    df["experience"] = df["experience"].fillna("zero")
    df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(df["test_score(out of 10)"].median())
    # Convert word2number
    from word2number import w2n
    df.experience = df.experience.apply(w2n.word_to_num)
    # Data Partition
    X = df[["experience","test_score(out of 10)", "interview_score(out of 10)"]]
    Y = df["salary($)"]
    model = LinearRegression()
    model.fit(X, Y)
    return model

model = load_model()
# Input sliders
st.sidebar.title("Hiring SDET")
person_exp = st.sidebar.number_input("Experience", min_value=0.0)
test_score = st.sidebar.number_input("test_score(out of 10)", min_value=0.0, max_value=10.0)
interview_score = st.sidebar.number_input("interview_score(out of 10)", min_value=0.0, max_value=10.0)

st.title("Hiring Salary Predictor using Machine Learning")
st.write("This Streamlit web app predicts the expected salary of a candidate based on their years of experience, test score, and interview performance.")
st.write("Built using a Linear Regression model trained on real hiring data, the app provides quick and accurate salary predictions to help hiring managers or job seekers make data-driven decisions.")
st.subheader("🧠 Powered by: Python, Pandas, Scikit-learn, and Streamlit")
# Predict button
if st.button("Predict the Salary"):
    input_data = np.array([[person_exp, test_score, interview_score]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
