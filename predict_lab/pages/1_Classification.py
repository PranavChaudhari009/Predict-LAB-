import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

# -----------------------------
# Load dataset
# -----------------------------
pd.read_csv(DATA_DIR / "loan_sanction_train.csv")

processed_data = preprocess_data(data)


# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_data(df):
    df = df.copy()

    # Drop Loan_ID
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Fill missing values
    mode_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "LoanAmount" in df.columns:
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    # Encode binary categorical columns
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    if "Married" in df.columns:
        df["Married"] = df["Married"].map({"Yes": 1, "No": 0})

    if "Education" in df.columns:
        df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})

    if "Self_Employed" in df.columns:
        df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})

    # Convert Dependents
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3)
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")
        df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])

    # One-hot encode Property_Area
    if "Property_Area" in df.columns:
        df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

    # Map Loan_Status
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].replace({"Y": 1, "N": 0})

    return df

# -----------------------------
# Preprocess dataset
# -----------------------------
processed_data = preprocess_data(data)

# -----------------------------
# Split features and target
# -----------------------------
X = processed_data.drop("Loan_Status", axis=1)
y = processed_data["Loan_Status"]

feature_columns = X.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Load models
# -----------------------------
joblib.load(MODELS_DIR / "classification_model.pkl")

joblib.load(REPORTS_DIR / "rf_model.joblib")
joblib.load(REPORTS_DIR / "lr_model.joblib")

# -----------------------------
# App title
# -----------------------------
st.title("🏦 Loan Approval Prediction")
st.write("Fill in the applicant details below to predict loan approval status.")

# -----------------------------
# Input layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    coapp_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
    loan_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0, step=1.0)
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

with col2:
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    app_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0, step=1.0)
    credit_history = st.selectbox("Credit History", [1, 0])

# -----------------------------
# Encode user input
# -----------------------------
gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

property_area_semiurban = 1 if property_area == "Semiurban" else 0
property_area_urban = 1 if property_area == "Urban" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Status"):
    user_data = pd.DataFrame([{
        "Gender": gender_val,
        "Married": married_val,
        "Dependents": dependents,
        "Education": education_val,
        "Self_Employed": self_employed_val,
        "ApplicantIncome": app_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area_Semiurban": property_area_semiurban,
        "Property_Area_Urban": property_area_urban
    }])

    user_data = user_data.reindex(columns=feature_columns, fill_value=0)

    prediction = prediction_model.predict(user_data)

    st.subheader("Prediction Result")

    pred_value = prediction[0]
    if pred_value == "Y":
        pred_value = 1
    elif pred_value == "N":
        pred_value = 0

    if pred_value == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    with st.expander("See Input Data Sent to Model"):
        st.dataframe(user_data)

# -----------------------------
# Model comparison
# -----------------------------
st.markdown("---")
st.title("Model Performance Comparison")

rf_pred = rf_model.predict(x_test)
lr_pred = lr_model.predict(x_test)

rf_pred = pd.Series(rf_pred).replace({"Y": 1, "N": 0})
lr_pred = pd.Series(lr_pred).replace({"Y": 1, "N": 0})
y_test = pd.Series(y_test).replace({"Y": 1, "N": 0})

rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.subheader("Logistic Regression")
    st.write("Accuracy:", round(lr_accuracy, 3))
    st.text("Classification Report")
    st.text(classification_report(y_test, lr_pred))

with comp_col2:
    st.subheader("Random Forest")
    st.write("Accuracy:", round(rf_accuracy, 3))
    st.text("Classification Report")
    st.text(classification_report(y_test, rf_pred))

if rf_accuracy > lr_accuracy:
    st.success(f"Best Model: Random Forest ({rf_accuracy:.3f})")
elif lr_accuracy > rf_accuracy:
    st.success(f"Best Model: Logistic Regression ({lr_accuracy:.3f})")
else:
    st.info(f"Both models have the same accuracy: {lr_accuracy:.3f}")
