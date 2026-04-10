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
    st.markdown(
        f"<h3 style='color:{color}; margin-bottom:0;'>{text}</h3>",
        unsafe_allow_html=True,
    )


def get_confidence(model, text: str) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])
        return float(np.max(proba[0]) * 100)

    if hasattr(model, "decision_function"):
        score = model.decision_function([text])
        score = np.ravel(score)

        if len(score) == 1:
            confidence = 1 / (1 + np.exp(-abs(score[0])))
            return float(confidence * 100)

        exp_scores = np.exp(score - np.max(score))
        probs = exp_scores / exp_scores.sum()
        return float(np.max(probs) * 100)

    return 0.0


def apply_news_rules(text: str, prediction: int):
    text_lower = text.lower()

    fake_patterns = [
        r"\bshocking\b",
        r"\bsecret\b",
        r"\bsecretly\b",
        r"\bhidden agenda\b",
        r"\bmind control\b",
        r"\bexperts warn\b",
        r"\bcures?\b",
        r"\bcompletely cures?\b",
        r"\bno medicine needed\b",
        r"\bmiracle cure\b",
        r"\bscientists confirm\b",
        r"\bscientists secretly confirmed\b",
        r"\bmedia is hiding\b",
        r"\b100%\b",
        r"\b100 percent\b",
        r"\bguaranteed\b",
    ]

    real_patterns = [
        r"\bgovernment\b",
        r"\bministry\b",
        r"\badministration\b",
        r"\bofficial\b",
        r"\breport\b",
        r"\bannounced\b",
        r"\baccording to\b",
        r"\bcommittee\b",
        r"\bdepartment\b",
        r"\bpolicy\b",
        r"\bstatement\b",
        r"\btransport\b",
        r"\bpublic\b",
        r"\bdata\b",
    ]

    fake_hits = sum(bool(re.search(pattern, text_lower)) for pattern in fake_patterns)
    real_hits = sum(bool(re.search(pattern, text_lower)) for pattern in real_patterns)

    if fake_hits >= 2:
        return 0, "fake_rule"

    if real_hits >= 2 and fake_hits == 0:
        return 1, "real_rule"

    return prediction, None


def ensure_file_exists(file_path: Path):
    if not file_path.exists():
        st.error(f"Missing file: {file_path}")
        st.stop()


# -----------------------------
# Data Loaders
# -----------------------------
@st.cache_data
def load_spam_data():
    spam_path = DATA_DIR / "spam.csv"
    ensure_file_exists(spam_path)

    data = pd.read_csv(spam_path, encoding="latin1")
    data.columns = data.columns.str.strip()

    # expected Kaggle-style columns v1, v2
    if "v1" not in data.columns or "v2" not in data.columns:
        st.error(f"spam.csv must contain columns 'v1' and 'v2'. Found: {data.columns.tolist()}")
        st.stop()

    data = data.rename(columns={"v1": "label", "v2": "message"})
    data = data[["label", "message"]].dropna().copy()
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)
    data["message"] = data["message"].astype(str).apply(clean_text)
    return data


def load_sentiment_data(sample_size: int = 40000):
    sentiment_path = DATA_DIR / "Twitter_Data.csv"
    ensure_file_exists(sentiment_path)

    data = pd.read_csv(sentiment_path)
    data.columns = data.columns.str.strip()

    st.write("Loaded sentiment file:", str(sentiment_path))
    st.write("Detected columns:", data.columns.tolist())

    text_col = None
    for col in data.columns:
        if col.lower() in ["clean_text", "text", "tweet", "message", "content"]:
            text_col = col
            break

    label_col = None
    for col in data.columns:
        if col.lower() in ["category", "label", "sentiment", "target", "polarity", "class"]:
            label_col = col
            break

    if text_col is None:
        st.error(f"Text column not found. Available columns: {data.columns.tolist()}")
        st.stop()

    if label_col is None:
        st.error(f"Label column not found. Available columns: {data.columns.tolist()}")
        st.stop()

    data = data.rename(columns={text_col: "clean_text", label_col: "category"})
    st.write("Columns after rename:", data.columns.tolist())

    # force fresh dataframe with only needed columns
    data = pd.DataFrame({
        "clean_text": data["clean_text"].fillna("").astype(str).apply(clean_text),
        "category": pd.to_numeric(data["category"], errors="coerce")
    })

    data = data.dropna(subset=["category"]).reset_index(drop=True)
    data["category"] = data["category"].round().astype(int)
    data = data[data["category"].isin([-1, 0, 1])].reset_index(drop=True)

    st.write("Final columns:", data.columns.tolist())
    st.write("Sample rows:", data.head())

    if "category" not in data.columns:
        st.error("category column missing after preprocessing")
        st.stop()

    if "clean_text" not in data.columns:
        st.error("clean_text column missing after preprocessing")
        st.stop()

    if data.empty:
        st.error("No valid rows found after filtering labels.")
        st.stop()

    if len(data) > sample_size:
        per_class = max(sample_size // 3, 1)
        data = (
            data.groupby("category", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), per_class), random_state=42))
            .reset_index(drop=True)
        )

    return data


def train_sentiment_models():
    data = load_sentiment_data()

    st.write("Inside train_sentiment_models columns:", data.columns.tolist())

    if "clean_text" not in data.columns or "category" not in data.columns:
        st.error(f"Columns missing in training step. Found: {data.columns.tolist()}")
        st.stop()

    X = data.loc[:, "clean_text"].copy()
    y = data.loc[:, "category"].copy()

    st.write("Target class counts:", y.value_counts().to_dict())

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": build_text_pipeline(LogisticRegression(max_iter=1000)),
        "Multinomial Naive Bayes": build_text_pipeline(MultinomialNB()),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
        }

    best_model_name = max(metrics, key=lambda name: metrics[name]["accuracy"])
    return models, metrics, best_model_name


@st.cache_data
def load_fake_news_data(sample_size: int = 30000):
    fake_path = DATA_DIR / "WELFake_sample.csv"
    ensure_file_exists(fake_path)

    data = pd.read_csv(
        fake_path,
        usecols=["title", "text", "label"],
        engine="python",
        on_bad_lines="skip",
    )
    data.columns = data.columns.str.strip()

    data = data.dropna(subset=["label"]).copy()
    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)
    data = data[data["label"].isin([0, 1])].copy()
    data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)

    if data.empty:
        st.error("No valid rows found in WELFake_sample.csv.")
        st.stop()

    if len(data) > sample_size:
        per_class = max(sample_size // 2, 1)
        data = (
            data.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), per_class), random_state=42))
            .reset_index(drop=True)
        )

    return data[["content", "label"]]


# -----------------------------
# Model Builders
# -----------------------------
def build_text_pipeline(model):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            ("clf", model),
        ]
    )


@st.cache_resource
def train_spam_models():
    data = load_spam_data()

    x_train, x_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    models = {
        "Multinomial Naive Bayes": build_text_pipeline(MultinomialNB()),
        "Logistic Regression": build_text_pipeline(LogisticRegression(max_iter=1000)),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

    best_model_name = max(metrics, key=lambda n: metrics[n]["accuracy"])
    return models, metrics, best_model_name


@st.cache_resource
def train_sentiment_models():
    data = load_sentiment_data()

    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["category"],
        test_size=0.2,
        random_state=42,
        stratify=data["category"],
    )

    models = {
        "Logistic Regression": build_text_pipeline(LogisticRegression(max_iter=1000)),
        "Multinomial Naive Bayes": build_text_pipeline(MultinomialNB()),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
        }

    best_model_name = max(metrics, key=lambda n: metrics[n]["accuracy"])
    return models, metrics, best_model_name


@st.cache_resource
def train_fake_news_models():
    data = load_fake_news_data()

    x_train, x_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    models = {
        "Logistic Regression": build_text_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Passive Aggressive": build_text_pipeline(
            PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        ),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "fake_f1": f1_score(y_test, preds, pos_label=0),
        }

    best_model_name = max(
        metrics, key=lambda n: (metrics[n]["fake_f1"], metrics[n]["accuracy"])
    )
    return models, metrics, best_model_name


# -----------------------------
# Example Inputs
# -----------------------------
SPAM_EXAMPLES = {
    "Spam Example": "Congratulations! You have won a free iPhone. Click the link now to claim your prize.",
    "Ham Example": "Hi, just checking if we are still meeting tomorrow at 11 AM for the project discussion.",
}

SENTIMENT_EXAMPLES = {
    "Positive Example": "This product is amazing, I absolutely loved the experience and would recommend it to everyone.",
    "Negative Example": "The service was terrible, very slow, and I am disappointed with the overall experience.",
    "Neutral Example": "The package arrived today and I have started using it for regular office work.",
}

FAKE_NEWS_EXAMPLES = {
    "Real News Example": (
        "The city administration announced a new public transport plan on Tuesday after reviewing "
        "traffic data, consulting urban development experts, and publishing an official statement "
        "through the transport department."
    ),
    "Fake News Example": (
        "Scientists secretly confirmed a miracle cure that works 100 percent in two days, "
        "and the media is hiding the truth from the public."
    ),
}


# -----------------------------
# UI
# -----------------------------
st.title("PredictLab NLP Studio")
st.write(
    "This page combines three NLP projects: spam email detection, sentiment analysis, and fake news detection."
)

selected_section = st.radio(
    "Choose NLP Project",
    ["Spam Email Detection", "Sentiment Analysis", "Fake News Detection"],
    horizontal=True,
)


# -----------------------------
# Spam Section
# -----------------------------
if selected_section == "Spam Email Detection":
    st.subheader("Spam Email Detection")
    st.write("Classify an email as spam or ham and compare two text classification models.")

    spam_models, spam_metrics, spam_best_model_name = train_spam_models()
    spam_model_choice = st.selectbox(
        "Select Spam Model",
        ["Multinomial Naive Bayes", "Logistic Regression"]
    )
    spam_selected_model = spam_models[spam_model_choice]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Spam Example"):
            st.session_state["spam_text_area"] = SPAM_EXAMPLES["Spam Example"]
    with col2:
        if st.button("Load Ham Example"):
            st.session_state["spam_text_area"] = SPAM_EXAMPLES["Ham Example"]

    spam_text = st.text_area("Enter Email Text", height=180, key="spam_text_area")

    if st.button("Predict Email Type"):
        if spam_text.strip():
            cleaned_text = clean_text(spam_text)
            pred = int(spam_selected_model.predict([cleaned_text])[0])
            confidence = get_confidence(spam_selected_model, cleaned_text)

            if pred == 1:
                render_label(f"Spam Email (Confidence: {confidence:.1f}%)", "red")
            else:
                render_label(f"Ham Email (Confidence: {confidence:.1f}%)", "green")

            st.info(f"Prediction made using: {spam_model_choice}")
        else:
            st.warning("Please enter email text first.")

    st.markdown("---")
    st.subheader("NLP Model Comparison & Evaluation")

    col1, col2 = st.columns(2)
    model_names = list(spam_metrics.keys())

    with col1:
        st.markdown(f"**{model_names[0]}**")
        st.write("Accuracy:", f"{spam_metrics[model_names[0]]['accuracy']:.3f}")
        st.write("F1 Score:", f"{spam_metrics[model_names[0]]['f1']:.3f}")

    with col2:
        st.markdown(f"**{model_names[1]}**")
        st.write("Accuracy:", f"{spam_metrics[model_names[1]]['accuracy']:.3f}")
        st.write("F1 Score:", f"{spam_metrics[model_names[1]]['f1']:.3f}")

    st.info(f"Best spam model: {spam_best_model_name}")
    st.write("Reason: Higher F1 score means better balance between precision and recall.")


# -----------------------------
# Sentiment Section
# -----------------------------
elif selected_section == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    st.write("Predict whether a comment is positive, negative, or neutral and compare two models.")

    sentiment_models, sentiment_metrics, sentiment_best_model_name = train_sentiment_models()
    sentiment_model_choice = st.selectbox(
        "Select Sentiment Model",
        ["Logistic Regression", "Multinomial Naive Bayes"]
    )
    sentiment_selected_model = sentiment_models[sentiment_model_choice]

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Load Positive Example"):
            st.session_state["sentiment_text_area"] = SENTIMENT_EXAMPLES["Positive Example"]
    with col2:
        if st.button("Load Negative Example"):
            st.session_state["sentiment_text_area"] = SENTIMENT_EXAMPLES["Negative Example"]
    with col3:
        if st.button("Load Neutral Example"):
            st.session_state["sentiment_text_area"] = SENTIMENT_EXAMPLES["Neutral Example"]

    sentiment_text = st.text_area("Enter Comment Text", height=180, key="sentiment_text_area")

    if st.button("Predict Sentiment"):
        if sentiment_text.strip():
            cleaned_text = clean_text(sentiment_text)
            pred = int(sentiment_selected_model.predict([cleaned_text])[0])
            confidence = get_confidence(sentiment_selected_model, cleaned_text)

            if pred == 1:
                render_label(f"Positive (Confidence: {confidence:.1f}%)", "green")
            elif pred == -1:
                render_label(f"Negative (Confidence: {confidence:.1f}%)", "red")
            else:
                render_label(f"Neutral (Confidence: {confidence:.1f}%)", "blue")

            st.info(f"Prediction made using: {sentiment_model_choice}")
        else:
            st.warning("Please enter comment text first.")

    st.markdown("---")
    st.subheader("NLP Model Comparison & Evaluation")

    col1, col2 = st.columns(2)
    model_names = list(sentiment_metrics.keys())

    with col1:
        st.markdown(f"**{model_names[0]}**")
        st.write("Accuracy:", f"{sentiment_metrics[model_names[0]]['accuracy']:.3f}")
        st.write("Macro F1:", f"{sentiment_metrics[model_names[0]]['macro_f1']:.3f}")

    with col2:
        st.markdown(f"**{model_names[1]}**")
        st.write("Accuracy:", f"{sentiment_metrics[model_names[1]]['accuracy']:.3f}")
        st.write("Macro F1:", f"{sentiment_metrics[model_names[1]]['macro_f1']:.3f}")

    st.info(f"Best sentiment model: {sentiment_best_model_name}")


# -----------------------------
# Fake News Section
# -----------------------------
else:
    st.subheader("Fake News Detection")
    st.write("Classify news text as real or fake and compare two fake-news detection models.")

    fake_models, fake_metrics, fake_best_model_name = train_fake_news_models()
    fake_model_choice = st.selectbox(
        "Select Fake News Model",
        ["Logistic Regression", "Passive Aggressive"]
    )
    fake_selected_model = fake_models[fake_model_choice]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Real News Example"):
            st.session_state["fake_text_area"] = FAKE_NEWS_EXAMPLES["Real News Example"]
    with col2:
        if st.button("Load Fake News Example"):
            st.session_state["fake_text_area"] = FAKE_NEWS_EXAMPLES["Fake News Example"]

    fake_text = st.text_area("Enter News Text", height=200, key="fake_text_area")

    if st.button("Predict News Type"):
        if fake_text.strip():
            cleaned_text = clean_text(fake_text)
            model_pred = int(fake_selected_model.predict([cleaned_text])[0])
            confidence = get_confidence(fake_selected_model, cleaned_text)

            final_pred, rule_used = apply_news_rules(fake_text, model_pred)

            if final_pred == 1:
                render_label(f"Real News (Confidence: {confidence:.1f}%)", "green")
            else:
                render_label(f"Fake News (Confidence: {confidence:.1f}%)", "red")

            st.info(f"Prediction made using: {fake_model_choice}")

            if rule_used == "fake_rule":
                st.warning("Safety rule applied: sensational fake-news patterns were detected.")
            elif rule_used == "real_rule":
                st.info("Real-news rule applied: official/news-report wording was detected.")
        else:
            st.warning("Please enter news text first.")

    st.markdown("---")
    st.subheader("NLP Model Comparison & Evaluation")

    col1, col2 = st.columns(2)
    model_names = list(fake_metrics.keys())

    with col1:
        st.markdown(f"**{model_names[0]}**")
        st.write("Accuracy:", f"{fake_metrics[model_names[0]]['accuracy']:.3f}")
        st.write("Fake F1:", f"{fake_metrics[model_names[0]]['fake_f1']:.3f}")

    with col2:
        st.markdown(f"**{model_names[1]}**")
        st.write("Accuracy:", f"{fake_metrics[model_names[1]]['accuracy']:.3f}")
        st.write("Fake F1:", f"{fake_metrics[model_names[1]]['fake_f1']:.3f}")

    st.info(f"Best fake-news model: {fake_best_model_name}")


# -----------------------------
# Graphs
# -----------------------------
st.markdown("---")
st.header("Model Performance Graphs")

if selected_section == "Spam Email Detection":
    st.subheader("Spam Model Accuracy and F1 Score")

    model_names = list(spam_metrics.keys())
    accuracies = [spam_metrics[name]["accuracy"] for name in model_names]
    f1_scores = [spam_metrics[name]["f1"] for name in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(model_names))
    width = 0.35

    ax.bar([i - width / 2 for i in x], accuracies, width=width, label="Accuracy", color="skyblue")
    ax.bar([i + width / 2 for i in x], f1_scores, width=width, label="F1 Score", color="orange")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Spam Detection Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)

elif selected_section == "Sentiment Analysis":
    st.subheader("Sentiment Model Accuracy and Macro F1")

    model_names = list(sentiment_metrics.keys())
    accuracies = [sentiment_metrics[name]["accuracy"] for name in model_names]
    macro_f1_scores = [sentiment_metrics[name]["macro_f1"] for name in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(model_names))
    width = 0.35

    ax.bar([i - width / 2 for i in x], accuracies, width=width, label="Accuracy", color="lightgreen")
    ax.bar([i + width / 2 for i in x], macro_f1_scores, width=width, label="Macro F1", color="salmon")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Sentiment Analysis Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)

else:
    st.subheader("Fake News Model Accuracy and Fake F1")

    model_names = list(fake_metrics.keys())
    accuracies = [fake_metrics[name]["accuracy"] for name in model_names]
    fake_f1_scores = [fake_metrics[name]["fake_f1"] for name in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(model_names))
    width = 0.35

    ax.bar([i - width / 2 for i in x], accuracies, width=width, label="Accuracy", color="cornflowerblue")
    ax.bar([i + width / 2 for i in x], fake_f1_scores, width=width, label="Fake F1", color="tomato")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Fake News Detection Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
