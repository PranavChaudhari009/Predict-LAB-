import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Time Series Analysis", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

st.title("Time Series Analysis")
st.write("Fill the following details to predict the next day's stock direction.")

# Load models once
with open(REPORTS_DIR / "lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open(REPORTS_DIR / "rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

model_choice = st.selectbox("Select Model", ["logistic regression", "random forest"])

# Load and prepare dataset for evaluation
df = pd.read_csv(DATA_DIR / "apple_stock.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").drop_duplicates()
df.set_index("Date", inplace=True)

# Feature engineering
df["Return_1D"] = df["Close"].pct_change()
df["Return_5D"] = df["Close"].pct_change(5)
df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_Ratio"] = df["SMA_5"] / df["SMA_10"]
df["Volatility_5D"] = df["Return_1D"].rolling(5).std()
df["Volume_Change"] = df["Volume"].pct_change()
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

feature_columns = [
    "Return_1D",
    "Return_5D",
    "SMA_Ratio",
    "Volatility_5D",
    "Volume_Change",
]

df = df.dropna(subset=feature_columns + ["Target"]).copy()

X = df[feature_columns]
y = df["Target"]

# Same evaluation style as training: chronological split
split_index = int(len(df) * 0.8)
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

st.subheader("Enter Input Values")

close_today = st.number_input("Today's Close Price", min_value=0.0)
close_yesterday = st.number_input("Yesterday's Close Price", min_value=0.0)
close_5_days_ago = st.number_input("Close Price 5 Days Ago", min_value=0.0)
sma_5 = st.number_input("5-Day Moving Average", min_value=0.0)
sma_10 = st.number_input("10-Day Moving Average", min_value=0.0)
volatility_5d = st.number_input("5-Day Volatility", min_value=0.0)
volume_today = st.number_input("Today's Volume", min_value=0.0)
volume_yesterday = st.number_input("Yesterday's Volume", min_value=0.0)

if st.button("Predict"):
    return_1d = (close_today - close_yesterday) / close_yesterday if close_yesterday != 0 else 0
    return_5d = (close_today - close_5_days_ago) / close_5_days_ago if close_5_days_ago != 0 else 0
    sma_ratio = sma_5 / sma_10 if sma_10 != 0 else 0
    volume_change = (volume_today - volume_yesterday) / volume_yesterday if volume_yesterday != 0 else 0

    input_data = pd.DataFrame([{
        "Return_1D": return_1d,
        "Return_5D": return_5d,
        "SMA_Ratio": sma_ratio,
        "Volatility_5D": volatility_5d,
        "Volume_Change": volume_change
    }])

    if model_choice == "logistic regression":
        model = lr_model
    else:
        model = rf_model

    prediction = model.predict(input_data)[0]

    st.write(f"Predicted Class: {prediction}")

    if prediction == 1:
        st.success("Prediction: Stock price may go UP tomorrow")
    else:
        st.error("Prediction: Stock price may go DOWN tomorrow")

    with st.expander("See Input Data Sent to Model"):
        st.dataframe(input_data)

st.markdown("---")
st.title("Model Performance Comparison")

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

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
