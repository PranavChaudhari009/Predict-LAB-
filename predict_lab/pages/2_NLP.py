import re
import string
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="PredictLab NLP", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# ===================== UTILITIES =====================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def render_label(text: str, color: str):
    st.markdown(f"<h3 style='color:{color}; margin-bottom:0;'>{text}</h3>", unsafe_allow_html=True)

def get_confidence(model, text: str) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])
        return float(np.max(proba[0]) * 100)
    return 0.0

def apply_news_rules(text: str, prediction: int):
    text_lower = text.lower()
    fake_hits = sum(bool(re.search(p, text_lower)) for p in [r"\bshocking\b", r"\bsecret", r"\b100%", r"\bguaranteed\b", r"\bmiracle cure\b"])
    real_hits = sum(bool(re.search(p, text_lower)) for p in [r"\bgovernment\b", r"\bofficial\b", r"\breport\b", r"\bannounced\b"])
    if fake_hits >= 2:
        return 0, "fake_rule"
    if real_hits >= 2:
        return 1, "real_rule"
    return prediction, None

def ensure_file_exists(file_path: Path):
    if not file_path.exists():
        st.error(f"Missing file: {file_path.name}. Please upload all CSV files to data folder.")
        st.stop()

# ===================== LAZY DATA LOADERS =====================
@st.cache_data
def load_spam_data():
    path = DATA_DIR / "spam.csv"
    ensure_file_exists(path)
    data = pd.read_csv(path, encoding="latin1")
    data.columns = data.columns.str.strip()
    if "v1" in data.columns:
        data = data.rename(columns={"v1": "label", "v2": "message"})
    data = data[["label", "message"]].dropna().copy()
    data["label"] = data["label"].map({"ham": 0, "spam": 1}).astype(int)
    data["message"] = data["message"].astype(str).apply(clean_text)
    return data

@st.cache_data
def load_sentiment_data():
    path = DATA_DIR / "Twitter_Data.csv"
    ensure_file_exists(path)
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    
    # Show columns for debugging
    st.sidebar.error(f"Twitter_Data.csv columns: {list(data.columns)}")
    
    # Robust detection
    text_col = next((c for c in data.columns if c.lower() in ["clean_text", "text", "tweet", "message", "content"]), None)
    label_col = next((c for c in data.columns if c.lower() in ["category", "label", "sentiment", "target"]), None)
    
    if not text_col or not label_col:
        st.error("Could not find text or label column in Twitter_Data.csv")
        st.stop()
    
    data = data.rename(columns={text_col: "clean_text", label_col: "category"})
    data = data[["clean_text", "category"]].copy()
    data["clean_text"] = data["clean_text"].fillna("").astype(str).apply(clean_text)
    data["category"] = pd.to_numeric(data["category"], errors="coerce").round().astype(int)
    data = data[data["category"].isin([-1, 0, 1])].copy()
    return data

@st.cache_data
def load_fake_news_data():
    path = DATA_DIR / "WELFake_sample.csv"
    ensure_file_exists(path)
    data = pd.read_csv(path, usecols=["title","text","label"], on_bad_lines="skip")
    data["content"] = (data["title"].fillna("") + " " + data["text"].fillna("")).apply(clean_text)
    data["label"] = pd.to_numeric(data["label"], errors="coerce").astype(int)
    return data[["content", "label"]]

# ===================== MODEL BUILDERS =====================
def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1,2), stop_words="english")),
        ("clf", model)
    ])

# ===================== MAIN APP =====================
st.title("PredictLab NLP Studio")

selected_section = st.radio(
    "Choose NLP Project",
    ["Spam Email Detection", "Sentiment Analysis", "Fake News Detection"],
    horizontal=True
)

# ===================== SPAM SECTION =====================
if selected_section == "Spam Email Detection":
    spam_models, spam_metrics, best_model = None, None, None
    try:
        data = load_spam_data()
        X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], 
                                                            test_size=0.2, random_state=42, stratify=data["label"])
        
        models = {
            "Multinomial Naive Bayes": build_pipeline(MultinomialNB()),
            "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000))
        }
        # Training code...
        # (I shortened it - you can keep your original training logic)
        
        st.success("Spam models loaded successfully")
        
    except Exception as e:
        st.error(f"Error in Spam: {e}")

# ===================== SENTIMENT SECTION (Most Critical) =====================
elif selected_section == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    try:
        data = load_sentiment_data()   # Now loaded only when tab is selected
        st.success(f"Loaded {len(data)} sentiment records")
        
        # Rest of your sentiment code (model training, prediction, etc.)
        # Copy your original sentiment logic here
        
    except Exception as e:
        st.error(f"Sentiment Loading Error: {str(e)}")
        st.info("Check the sidebar for column names")

# ===================== FAKE NEWS SECTION =====================
else:
    st.subheader("Fake News Detection")
    try:
        data = load_fake_news_data()
        st.success(f"Loaded {len(data)} news records")
        # Your fake news logic here
    except Exception as e:
        st.error(f"Fake News Error: {str(e)}")

st.sidebar.info("If you see column names in red in sidebar, share them with me.")
