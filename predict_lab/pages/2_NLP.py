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

# -----------------------------
# Utility Functions
# -----------------------------
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
    if hasattr(model, "decision_function"):
        score = np.ravel(model.decision_function([text]))
        if len(score) == 1:
            return float(1 / (1 + np.exp(-abs(score[0]))) * 100)
        exp_scores = np.exp(score - np.max(score))
        return float(np.max(exp_scores / exp_scores.sum()) * 100)
    return 0.0

def apply_news_rules(text: str, prediction: int):
    text_lower = text.lower()
    fake_patterns = [r"\bshocking\b", r"\bsecret(ly)?\b", r"\bhidden agenda\b", r"\bmind control\b",
                     r"\bexperts warn\b", r"\bcures?\b", r"\bmiracle cure\b", r"\b100%\b", r"\bguaranteed\b"]
    real_patterns = [r"\bgovernment\b", r"\bministry\b", r"\bofficial\b", r"\breport\b", r"\bannounced\b",
                     r"\baccording to\b", r"\bcommittee\b", r"\bdepartment\b", r"\bpolicy\b"]
    
    fake_hits = sum(bool(re.search(p, text_lower)) for p in fake_patterns)
    real_hits = sum(bool(re.search(p, text_lower)) for p in real_patterns)
    
    if fake_hits >= 2:
        return 0, "fake_rule"
    if real_hits >= 2 and fake_hits == 0:
        return 1, "real_rule"
    return prediction, None

def ensure_file_exists(file_path: Path):
    if not file_path.exists():
        st.error(f"❌ Missing file: {file_path.name}")
        st.stop()

# -----------------------------
# Improved Data Loaders with Heavy Debugging
# -----------------------------
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
def load_sentiment_data(sample_size: int = 40000):
    path = DATA_DIR / "Twitter_Data.csv"
    ensure_file_exists(path)
    
    data = pd.read_csv(path, low_memory=False)
    original_columns = list(data.columns)
    data.columns = data.columns.str.strip()
    
    st.sidebar.info(f"**Debug Info** - Twitter_Data.csv columns: {original_columns}")
    
    # Very robust column detection
    text_col = None
    for col in data.columns:
        if col.lower() in ["clean_text", "text", "tweet", "message", "content", "body", "tweet_text"]:
            text_col = col
            break
    
    label_col = None
    for col in data.columns:
        if col.lower() in ["category", "label", "sentiment", "target", "polarity", "class", "score", "sentiments"]:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        st.error(f"""
        🚨 **Column Detection Failed in Twitter_Data.csv**
        
        **Found columns:** {original_columns}
        
        Please make sure your CSV has:
        - A text column (e.g., clean_text, text, tweet)
        - A label column (e.g., category, label, sentiment)
        """)
        st.stop()
    
    # Rename and clean
    data = data.rename(columns={text_col: "clean_text", label_col: "category"})
    data = data[["clean_text", "category"]].copy()
    
    data["clean_text"] = data["clean_text"].fillna("").astype(str).apply(clean_text)
    data["category"] = pd.to_numeric(data["category"], errors="coerce")
    data = data.dropna(subset=["category"]).copy()
    data["category"] = data["category"].round().astype(int)
    data = data[data["category"].isin([-1, 0, 1])].copy()
    
    if len(data) > sample_size:
        per_class = max(sample_size // 3, 1)
        data = data.groupby("category", group_keys=False).apply(
            lambda x: x.sample(min(len(x), per_class), random_state=42)
        ).reset_index(drop=True)
    
    return data

@st.cache_data
def load_fake_news_data(sample_size: int = 30000):
    path = DATA_DIR / "WELFake_sample.csv"
    ensure_file_exists(path)
    data = pd.read_csv(path, usecols=["title", "text", "label"], on_bad_lines="skip")
    data.columns = data.columns.str.strip()
    data["content"] = (data["title"].fillna("") + " " + data["text"].fillna("")).apply(clean_text)
    data["label"] = pd.to_numeric(data["label"], errors="coerce").astype(int)
    data = data[data["label"].isin([0, 1])].copy()
    
    if len(data) > sample_size:
        per_class = max(sample_size // 2, 1)
        data = data.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), per_class), random_state=42)
        ).reset_index(drop=True)
    return data[["content", "label"]]

# -----------------------------
# Model Training Functions
# -----------------------------
def build_text_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                                  max_df=0.95, sublinear_tf=True, stop_words="english")),
        ("clf", model)
    ])

@st.cache_resource
def train_spam_models():
    data = load_spam_data()
    X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"],
                                                        test_size=0.2, random_state=42, stratify=data["label"])
    models = {
        "Multinomial Naive Bayes": build_text_pipeline(MultinomialNB()),
        "Logistic Regression": build_text_pipeline(LogisticRegression(max_iter=1000))
    }
    # ... (rest same as before)
    metrics = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        metrics[name] = {"accuracy": accuracy_score(y_test, p), "f1": f1_score(y_test, p)}
    best = max(metrics, key=lambda x: metrics[x]["accuracy"])
    return models, metrics, best

@st.cache_resource
def train_sentiment_models():
    data = load_sentiment_data()
    X_train, X_test, y_train, y_test = train_test_split(data["clean_text"], data["category"],
                                                        test_size=0.2, random_state=42, stratify=data["category"])
    models = {
        "Logistic Regression": build_text_pipeline(LogisticRegression(max_iter=1000)),
        "Multinomial Naive Bayes": build_text_pipeline(MultinomialNB())
    }
    metrics = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, p),
            "macro_f1": f1_score(y_test, p, average="macro")
        }
    best = max(metrics, key=lambda x: metrics[x]["accuracy"])
    return models, metrics, best

@st.cache_resource
def train_fake_news_models():
    data = load_fake_news_data()
    X_train, X_test, y_train, y_test = train_test_split(data["content"], data["label"],
                                                        test_size=0.2, random_state=42, stratify=data["label"])
    models = {
        "Logistic Regression": build_text_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Passive Aggressive": build_text_pipeline(PassiveAggressiveClassifier(max_iter=1000, random_state=42))
    }
    metrics = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, p),
            "fake_f1": f1_score(y_test, p, pos_label=0)
        }
    best = max(metrics, key=lambda x: (metrics[x]["fake_f1"], metrics[x]["accuracy"]))
    return models, metrics, best

# -----------------------------
# Rest of the UI code (same as my previous version)
# -----------------------------
# ... [I am keeping it short here - use the full UI code from my previous response]

st.title("PredictLab NLP Studio")
st.write("Spam Detection • Sentiment Analysis • Fake News Detection")

selected_section = st.radio("Choose NLP Project", 
    ["Spam Email Detection", "Sentiment Analysis", "Fake News Detection"], horizontal=True)

# Add this at the very top after title to see debug info
if "debug_shown" not in st.session_state:
    st.info("App is trying to load data... Check sidebar for column info if Sentiment tab fails.")
    st.session_state.debug_shown = True

# Then continue with your if-elif sections (use the cleaned version from previous message)

# For brevity, please copy the full UI + Graph part from my previous response.
