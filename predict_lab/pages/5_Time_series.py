import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Time Series Analysis", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

st.title("Apple Stock Price Direction Prediction(Time Series Analysis)")
st.write("Fill the following details to predict the next day's stock direction.")

# Load models once
with open(REPORTS_DIR / "lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open(REPORTS_DIR / "rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest"]
)

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

# Chronological split
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

    # Use selected model
    if model_choice == "Logistic Regression":
        model = lr_model
    else:
        model = rf_model

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    # Banner like screenshot
    st.markdown(
        f"""
        <div style="
            background-color:#1e3348;
            padding:12px 13px;
            border-radius:5px;
            margin-bottom:15px;
        ">
            <h6 style="color:#3399ff; margin:0;">
                Prediction made using: {model_choice}
            </h6>
        </div>
        """,
        unsafe_allow_html=True
    )

    

    if prediction == 1:
        st.success(f"{model_choice} Prediction: Stock price may go UP tomorrow")
    else:
        st.error(f"{model_choice} Prediction: Stock price may go DOWN tomorrow")

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


st.header("Data Visualization")



plot_df = df.iloc[split_index:].copy()
plot_df["Actual"] = y_test.values
plot_df["LR_Pred"] = lr_pred
plot_df["RF_Pred"] = rf_pred

st.subheader("Logistic Regression Prediction vs Actual")
fig6, ax6 = plt.subplots(figsize=(12, 4))
ax6.plot(plot_df.index, plot_df["Actual"], label="Actual", color="black")
ax6.plot(plot_df.index, plot_df["LR_Pred"], label="LR Prediction", color="blue", alpha=0.7)
ax6.set_title("Logistic Regression vs Actual")
ax6.set_xlabel("Date")
ax6.set_ylabel("Direction")
ax6.legend()
ax6.grid(True, alpha=0.3)
st.pyplot(fig6)

st.subheader("Random Forest Prediction vs Actual")
fig7, ax7 = plt.subplots(figsize=(12, 4))
ax7.plot(plot_df.index, plot_df["Actual"], label="Actual", color="black")
ax7.plot(plot_df.index, plot_df["RF_Pred"], label="RF Prediction", color="green", alpha=0.7)
ax7.set_title("Random Forest vs Actual")
ax7.set_xlabel("Date")
ax7.set_ylabel("Direction")
ax7.legend()
ax7.grid(True, alpha=0.3)
st.pyplot(fig7)

st.subheader("Logistic Regression vs Random Forest Predictions")
fig8, ax8 = plt.subplots(figsize=(12, 4))
ax8.plot(plot_df.index, plot_df["LR_Pred"], label="Logistic Regression", color="blue")
ax8.plot(plot_df.index, plot_df["RF_Pred"], label="Random Forest", color="green")
ax8.set_title("Model Prediction Comparison")
ax8.set_xlabel("Date")
ax8.set_ylabel("Predicted Direction")
ax8.legend()
ax8.grid(True, alpha=0.3)
st.pyplot(fig8)

st.subheader("Model Accuracy Comparison")
fig9, ax9 = plt.subplots(figsize=(8, 5))
models = ["Logistic Regression", "Random Forest"]
accuracies = [lr_accuracy, rf_accuracy]
colors = ["blue", "green"]
ax9.bar(models, accuracies, color=colors)
ax9.set_title("Accuracy Comparison of Models")
ax9.set_ylabel("Accuracy")
ax9.set_ylim(0, 1)
for i, v in enumerate(accuracies):
    ax9.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
st.pyplot(fig9)

