import streamlit as st
import pandas as pd
import joblib
import numpy as np

kmeans = joblib.load("user_segmentation_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("User Profiling & Segmentation Dashboard")

st.sidebar.header("Enter User Data")
age = st.sidebar.selectbox("Age Group", ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
device = st.sidebar.selectbox("Device Usage", ['Mobile', 'Desktop', 'Tablet'])
income = st.sidebar.selectbox("Income Level", ['Low', 'Medium', 'High'])
likes = st.sidebar.number_input("Likes and Reactions", min_value=0, max_value=1000, value=100)
followed = st.sidebar.number_input("Followed Accounts", min_value=0, max_value=1000, value=50)
weekday_hours = st.sidebar.number_input("Time Spent Online (hrs/weekday)", min_value=0.0, max_value=24.0, value=2.0)
weekend_hours = st.sidebar.number_input("Time Spent Online (hrs/weekend)", min_value=0.0, max_value=24.0, value=3.0)
ctr = st.sidebar.number_input("Click-Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.1)
conversion = st.sidebar.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.05)
ad_time = st.sidebar.number_input("Ad Interaction Time (sec)", min_value=0, max_value=3600, value=30)

age_mapping = {'18-24':21, '25-34':30, '35-44':40, '45-54':50, '55-64':60, '65+':70}
age_numeric = age_mapping[age]

user_input = pd.DataFrame({
    'Age_numeric':[age_numeric],
    'Likes and Reactions':[likes],
    'Followed Accounts':[followed],
    'Time Spent Online (hrs/weekday)':[weekday_hours],
    'Time Spent Online (hrs/weekend)':[weekend_hours],
    'Click-Through Rates (CTR)':[ctr],
    'Conversion Rates':[conversion],
    'Ad Interaction Time (sec)':[ad_time],
    'Gender':[gender],
    'Device Usage':[device],
    'Income Level':[income]
})

user_scaled = preprocessor.transform(user_input)
segment = kmeans.predict(user_scaled)[0]

st.subheader("Predicted User Segment")
st.write(f"The user belongs to **Segment {segment}**")

st.subheader("Segment Profile (Average Values)")
segment_profile = pd.read_csv("segment_profile.csv") 