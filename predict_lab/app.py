import streamlit as st

st.set_page_config(page_title="PredictLab", layout="wide")

st.title("PredictLab – ML Prediction Platform")

st.write("""
Welcome to PredictLab.

This platform demonstrates multiple machine learning workflows:
st.header(Classification)
         Loan Approval Prediction 
- Regression
         House Price Prediction 
- NLP
         1-Email Spam Classifer          
         2-Comment Sentimental Analysis
         3-Fake News Detection 
- Clustering

- Time Series
         Stock Price Movement 
- Recommedation system 
         E-commerce Product Recommedations
""")

st.info("Use the sidebar to navigate between modules.")
