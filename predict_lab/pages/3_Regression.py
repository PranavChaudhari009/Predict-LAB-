import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="House Price Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "house_price_dataset_india_12k.csv"

TARGET_COLUMN = "Market_Price_INR"
DROP_COLUMNS = ["House_ID", "Market_Price_INR", "Price_per_sqft_INR"]
CATEGORICAL_COLUMNS = ["City", "Locality_Tier", "Furnishing"]


@st.cache_data
def load_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_training_data(df: pd.DataFrame):
    encoded_df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    x = encoded_df.drop(DROP_COLUMNS, axis=1)
    y = encoded_df[TARGET_COLUMN]
    return x, y


def compute_metrics(model, x_test, y_test):
    predictions = model.predict(x_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }


@st.cache_resource
def train_models(df: pd.DataFrame):
    x, y = prepare_training_data(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    lr_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)

    lr_metrics = compute_metrics(lr_model, x_test, y_test)
    rf_metrics = compute_metrics(rf_model, x_test, y_test)

    return {
        "x_columns": x.columns.tolist(),
        "lr_model": lr_model,
        "rf_model": rf_model,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
    }


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        "Dataset not found. Add `house_price_dataset_india_12k.csv` to the `data` folder and reload the app."
    )
    st.stop()

st.title("House Price Prediction")
st.write("Fill in the property details below to predict the estimated house price.")

training_artifacts = train_models(df)
x_columns = training_artifacts["x_columns"]
lr_model = training_artifacts["lr_model"]
rf_model = training_artifacts["rf_model"]
lr_metrics = training_artifacts["lr_metrics"]
rf_metrics = training_artifacts["rf_metrics"]

model_choice = st.selectbox(
    "Select Prediction Model",
    ["Random Forest", "Linear Regression"]
)

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", sorted(df["City"].dropna().unique()))
    locality_tier = st.selectbox("Locality Tier", sorted(df["Locality_Tier"].dropna().unique()))
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=3, step=1)
    bathroom = st.number_input("Bathroom", min_value=1, max_value=10, value=2, step=1)
    super_area = st.number_input("Super Area", min_value=100.0, value=1200.0, step=10.0)
    carpet_area = st.number_input("Carpet Area", min_value=100.0, value=950.0, step=10.0)
    floor_no = st.number_input("Floor No", min_value=0, max_value=100, value=5, step=1)
    total_floor = st.number_input("Total Floor", min_value=1, max_value=100, value=12, step=1)
    property_age = st.number_input("Property Age", min_value=0, max_value=100, value=8, step=1)

with col2:
    parking = st.selectbox("Parking", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    furnishing = st.selectbox("Furnishing", sorted(df["Furnishing"].dropna().unique()))
    lift = st.selectbox("Lift", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    gated_society = st.selectbox("Gated Society", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    distance_to_metro_km = st.number_input("Distance to Metro (km)", min_value=0.0, value=2.5, step=0.1)
    distance_to_citycenter_km = st.number_input("Distance to City Center (km)", min_value=0.0, value=8.0, step=0.1)
    nearby_school_km = st.number_input("Nearby School (km)", min_value=0.0, value=1.2, step=0.1)
    nearby_hospital_km = st.number_input("Nearby Hospital (km)", min_value=0.0, value=2.1, step=0.1)
    crime_rate_index = st.number_input("Crime Rate Index", min_value=0.0, value=28.0, step=0.1)

if st.button("Predict House Price"):
    user_input = {
        "City": city,
        "Locality_Tier": locality_tier,
        "BHK": bhk,
        "Bathroom": bathroom,
        "Super_Area": super_area,
        "Carpet_Area": carpet_area,
        "Floor_No": floor_no,
        "Total_Floor": total_floor,
        "Property_Age": property_age,
        "Parking": parking,
        "Furnishing": furnishing,
        "Lift": lift,
        "Gated_Society": gated_society,
        "Distance_to_Metro_km": distance_to_metro_km,
        "Distance_to_CityCenter_km": distance_to_citycenter_km,
        "Nearby_School_km": nearby_school_km,
        "Nearby_Hospital_km": nearby_hospital_km,
        "Crime_Rate_Index": crime_rate_index,
    }

    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    user_df = user_df.reindex(columns=x_columns, fill_value=0)

    if model_choice == "Random Forest":
        selected_model = rf_model
    else:
        selected_model = lr_model

    predicted_price = selected_model.predict(user_df)[0]

    st.subheader("Prediction Result")
    st.success(f"Estimated House Price: INR {predicted_price:,.2f}")
    st.info(f"Prediction made using: {model_choice}")

    with st.expander("See input data sent to model"):
        st.dataframe(user_df)

st.markdown("---")
st.title("Model Performance Comparison")

compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    st.subheader("Linear Regression")
    st.write("MAE:", f"{lr_metrics['MAE']:,.2f}")
    st.write("RMSE:", f"{lr_metrics['RMSE']:,.2f}")
    st.write("R2:", f"{lr_metrics['R2']:.3f}")

with compare_col2:
    st.subheader("Random Forest")
    st.write("MAE:", f"{rf_metrics['MAE']:,.2f}")
    st.write("RMSE:", f"{rf_metrics['RMSE']:,.2f}")
    st.write("R2:", f"{rf_metrics['R2']:.3f}")

if rf_metrics["R2"] > lr_metrics["R2"]:
    st.success(f"Best Model: Random Forest ({rf_metrics['R2']:.3f})")
elif lr_metrics["R2"] > rf_metrics["R2"]:
    st.success(f"Best Model: Linear Regression ({lr_metrics['R2']:.3f})")
else:
    st.info(f"Both models have the same R2 score: {lr_metrics['R2']:.3f}")


col1, col2 = st.columns(2)

with col1:
    st.subheader("Linear Regression")
    st.image(
        str(Path(__file__).resolve().parents[1] / "images" / "avsp lr.png"),
        caption="Actual Values vs Prediction",
        width=500
    )
    st.image(
        str(Path(__file__).resolve().parents[1] / "images" / "pvsr lr.png"),
        caption="Residual",
        width=500
    )

with col2:
    st.subheader("Random Forest Regressor")
    st.image(
        str(Path(__file__).resolve().parents[1] / "images" / "avsp rfr.png"),
        caption="Actual Values vs Prediction",
        width=480
    )
    st.image(
        str(Path(__file__).resolve().parents[1] / "images" / "pvsr rfr.png"),
        caption="Residual",
        width=500
    )

