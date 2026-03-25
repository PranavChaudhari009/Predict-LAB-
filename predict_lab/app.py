import streamlit as st

st.set_page_config(page_title="PredictLab", layout="wide")

st.title("PredictLab – ML Prediction Platform")

st.write("""
Welcome to PredictLab.

This platform demonstrates multiple machine learning workflows:
""")

# Create 2 columns
col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:
    st.header("Classification")
    st.write("Loan Approval Prediction")

    st.header("NLP")
    st.markdown("""
    - Email Spam Classifier  
    - Comment Sentiment Analysis  
    - Fake News Detection  
    """)

    st.header("Time Series")
    st.write("Stock Price Movement")
st.write("-----------------------------------------------------------------------------------------------------------------------")
# RIGHT COLUMN
with col2:
    st.header("Regression")
    st.write("House Price Prediction")

    st.header("Clustering")

    st.header("Recommendation System")
    st.write("E-commerce Product Recommendations")

st.info("Use the sidebar to navigate between modules.")
