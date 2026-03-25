import streamlit as st

st.set_page_config(page_title="PredictLab", layout="wide")

st.title("PredictLab – ML Prediction Platform")

st.write("""
Welcome to PredictLab.

This platform demonstrates multiple machine learning workflows:
""")
st.header("Classification"),
st.sudheader("Loan Approval Prediction")

st.header("Regression")
st.subheader("House Price Prediction")
         
st.header("NLP")
st.write("1. Email Spam Classifier")
st.write("2. Comment Sentiment Analysis")
st.write("3. Fake News Detection")

st.header("Clustering")
st.subheader("Customer Chrun")

st.header("Time Series")
st.subheader("Stock Price Movement ")

st.header("Recommedation system ")
st.subheader("E-commerce Product Recommedations")


st.info("Use the sidebar to navigate between modules.")
