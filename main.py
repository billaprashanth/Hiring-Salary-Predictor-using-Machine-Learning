import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
@st.cache_data
def load_model():
    # Used relative path
    file_path = os.path.join(os.path.dirname(__file__), "hiring.csv")
    # Load and display data 
    df = pd.read_csv(file_path)
    # Filling the nan values
    df["experience"] = df["experience"].fillna("zero")
    df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(df["test_score(out of 10)"].median())
    # Convert word2number 
    from word2number import w2n
    df.experience = df.experience.apply(w2n.word_to_num)
    st.write("Loaded Data", df.head())
    # Data Partition
    X = df[["experience","test_score(out of 10)", "interview_score(out of 10)"]]
    Y = df["salary($)"]
    model = LinearRegression()
    model.fit(X, Y)
    return model

model = load_model()
# Input sliders
st.sidebar.title("Hiring SDET")
person_exp = st.sidebar.slider("Experience", min_value=0.0, max_value=10.0)
test_score = st.sidebar.slider("test_score(out of 10)", min_value=0.0, max_value=10.0)
interview_score = st.sidebar.slider("interview_score(out of 10)", min_value=0.0, max_value=10.0)

st.title("Hiring Salary Predictor using Machine Learning")
st.write("""
This app predicts the **expected salary** of a candidate based on:
- Years of experience  
- Test score  
- Interview performance  
Built using 🧠 **Linear Regression**, **pandas**, and **Streamlit**.
""")    
# Predict button
if st.button("Predict the Salary"):
    input_data = np.array([[person_exp, test_score, interview_score]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
