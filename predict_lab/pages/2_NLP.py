import re
import string

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"



st.set_page_config(page_title="PredictLab NLP", layout="wide")


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


def apply_fake_news_rules(text: str, prediction: int):
    fake_patterns = [
        r"\bshocking\b",
        r"\bsecret\b",
        r"\bsecretly\b",
        r"\bhidden agenda\b",
        r"\bmind control\b",
        r"\bexperts warn\b",
        r"!{2,}",
        r"\bcures?\b",
        r"\bcompletely cures?\b",
        r"\bno medicine needed\b",
        r"\bmiracle cure\b",
        r"\bscientists confirm\b",
        r"\bscientists secretly confirmed\b",
        r"\bmedia is hiding\b",
        r"\bfor \d+ days\b",
        r"\b100%\b",
        r"\b100 percent\b",
        r"\bguaranteed\b",
    ]

    fake_hits = sum(bool(re.search(pattern, text.lower())) for pattern in fake_patterns)
    if fake_hits >= 2:
        return 0, True
    return prediction, False


@st.cache_data
def load_spam_data():
    pd.read_csv(DATA_DIR / "spam.csv", ...)

    data = data.rename(columns={"v1": "label", "v2": "message"})
    data = data[["label", "message"]].dropna().copy()
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    data["message"] = data["message"].astype(str).apply(clean_text)
    return data


@st.cache_data
def load_sentiment_data(sample_size: int = 40000):
    pd.read_csv(DATA_DIR / "Twitter_Data.csv", ...)

    data["clean_text"] = data["clean_text"].fillna("").astype(str).apply(clean_text)
    data["category"] = pd.to_numeric(data["category"], errors="coerce")
    data = data.dropna(subset=["category"]).copy()
    data["category"] = data["category"].round().astype(int)
    data = data[data["category"].isin([-1, 0, 1])].copy()

    if len(data) > sample_size:
        data = (
            data.groupby("category", group_keys=False)
            .apply(lambda frame: frame.sample(min(len(frame), sample_size // 3), random_state=42))
            .reset_index(drop=True)
        )

    return data


@st.cache_data
def load_fake_news_data(sample_size: int = 30000):
    pd.read_csv(DATA_DIR / "WELFake_Dataset.csv",

        usecols=["title", "text", "label"],
        engine="python",
        on_bad_lines="skip",
    )
    data = data.dropna(subset=["label"]).copy()
    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)
    data = data[data["label"].isin([0, 1])].copy()
    data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)

    if len(data) > sample_size:
        data = (
            data.groupby("label", group_keys=False)
            .apply(lambda frame: frame.sample(min(len(frame), sample_size // 2), random_state=42))
            .reset_index(drop=True)
        )

    return data[["content", "label"]]


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
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
        }

    best_model_name = max(metrics, key=lambda name: metrics[name]["accuracy"])
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
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
        }

    best_model_name = max(metrics, key=lambda name: metrics[name]["accuracy"])
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
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "fake_f1": f1_score(y_test, predictions, pos_label=0),
        }

    best_model_name = max(
        metrics, key=lambda name: (metrics[name]["fake_f1"], metrics[name]["accuracy"])
    )
    return models, metrics, best_model_name


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
    "Real News Example": "The city administration announced a new public transport plan after reviewing traffic data and consulting urban development experts.",
    "Fake News Example": "Scientists secretly confirmed a miracle cure that works 100 percent in two days, but the media is hiding the truth.",
}


st.title("PredictLab NLP Studio")
st.write(
    "This page combines three NLP projects: spam email detection, sentiment analysis, and fake news detection."
)

selected_section = st.radio(
    "Choose NLP Project",
    ["Spam Email Detection", "Sentiment Analysis", "Fake News Detection"],
    horizontal=True,
)

if selected_section == "Spam Email Detection":
    st.subheader("Spam Email Detection")
    st.write("Classify an email as spam or ham and compare two text classification models.")

    spam_models, spam_metrics, spam_best_model_name = train_spam_models()
    spam_best_model = spam_models[spam_best_model_name]

    spam_btn_col1, spam_btn_col2 = st.columns(2)
    with spam_btn_col1:
        if st.button("Load Spam Example", key="spam_example_btn"):
            st.session_state["spam_text_area"] = SPAM_EXAMPLES["Spam Example"]
    with spam_btn_col2:
        if st.button("Load Ham Example", key="ham_example_btn"):
            st.session_state["spam_text_area"] = SPAM_EXAMPLES["Ham Example"]

    spam_text = st.text_area("Enter Email Text", height=180, key="spam_text_area")

    if st.button("Predict Email Type", key="predict_spam"):
        spam_prediction = int(spam_best_model.predict([clean_text(spam_text)])[0])
        if spam_prediction == 1:
            render_label("Spam Email", "red")
        else:
            render_label("Ham Email", "green")

    st.markdown("---")
    st.subheader("Model Performance Comparison")
    spam_metric_col1, spam_metric_col2 = st.columns(2)
    model_names = list(spam_metrics.keys())

    with spam_metric_col1:
        st.markdown(f"**{model_names[0]}**")
        st.write("Accuracy:", f"{spam_metrics[model_names[0]]['accuracy']:.3f}")
        st.write("F1 Score:", f"{spam_metrics[model_names[0]]['f1']:.3f}")

    with spam_metric_col2:
        st.markdown(f"**{model_names[1]}**")
        st.write("Accuracy:", f"{spam_metrics[model_names[1]]['accuracy']:.3f}")
        st.write("F1 Score:", f"{spam_metrics[model_names[1]]['f1']:.3f}")

    st.info(f"Best spam model: {spam_best_model_name}")

elif selected_section == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    st.write("Predict whether a comment is positive, negative, or neutral and compare two models.")

    sentiment_models, sentiment_metrics, sentiment_best_model_name = train_sentiment_models()
    sentiment_best_model = sentiment_models[sentiment_best_model_name]

    sent_btn_col1, sent_btn_col2, sent_btn_col3 = st.columns(3)
    with sent_btn_col1:
        if st.button("Load Positive Example", key="positive_example_btn"):
            st.session_state["sentiment_text_area"] = SENTIMENT_EXAMPLES["Positive Example"]
    with sent_btn_col2:
        if st.button("Load Negative Example", key="negative_example_btn"):
            st.session_state["sentiment_text_area"] = SENTIMENT_EXAMPLES["Negative Example"]
    with sent_btn_col3:
        if st.button("Load Neutral Example", key="neutral_example_btn"):
            st.session_state["sentiment_text_area"] = SENTIMENT_EXAMPLES["Neutral Example"]

    sentiment_text = st.text_area("Enter Comment Text", height=180, key="sentiment_text_area")

    if st.button("Predict Sentiment", key="predict_sentiment"):
        sentiment_prediction = int(sentiment_best_model.predict([clean_text(sentiment_text)])[0])
        if sentiment_prediction == 1:
            render_label("Positive", "green")
        elif sentiment_prediction == -1:
            render_label("Negative", "red")
        else:
            render_label("Neutral", "blue")

    st.markdown("---")
    st.subheader("Model Performance Comparison")
    sentiment_metric_col1, sentiment_metric_col2 = st.columns(2)
    sentiment_model_names = list(sentiment_metrics.keys())

    with sentiment_metric_col1:
        st.markdown(f"**{sentiment_model_names[0]}**")
        st.write("Accuracy:", f"{sentiment_metrics[sentiment_model_names[0]]['accuracy']:.3f}")
        st.write("Macro F1:", f"{sentiment_metrics[sentiment_model_names[0]]['macro_f1']:.3f}")

    with sentiment_metric_col2:
        st.markdown(f"**{sentiment_model_names[1]}**")
        st.write("Accuracy:", f"{sentiment_metrics[sentiment_model_names[1]]['accuracy']:.3f}")
        st.write("Macro F1:", f"{sentiment_metrics[sentiment_model_names[1]]['macro_f1']:.3f}")

    st.info(f"Best sentiment model: {sentiment_best_model_name}")

else:
    st.subheader("Fake News Detection")
    st.write("Classify news text as real or fake and compare two fake-news detection models.")

    fake_models, fake_metrics, fake_best_model_name = train_fake_news_models()
    fake_best_model = fake_models[fake_best_model_name]

    fake_btn_col1, fake_btn_col2 = st.columns(2)
    with fake_btn_col1:
        if st.button("Load Real News Example", key="real_news_btn"):
            st.session_state["fake_text_area"] = FAKE_NEWS_EXAMPLES["Real News Example"]
    with fake_btn_col2:
        if st.button("Load Fake News Example", key="fake_news_btn"):
            st.session_state["fake_text_area"] = FAKE_NEWS_EXAMPLES["Fake News Example"]

    fake_text = st.text_area("Enter News Text", height=200, key="fake_text_area")

    if st.button("Predict News Type", key="predict_fake_news"):
        fake_prediction = int(fake_best_model.predict([clean_text(fake_text)])[0])
        fake_prediction, rule_applied = apply_fake_news_rules(fake_text, fake_prediction)
        if fake_prediction == 1:
            render_label("Real News", "green")
        else:
            render_label("Fake News", "red")
        if rule_applied:
            st.warning("Safety rule applied: sensational fake-news patterns were detected.")

    st.markdown("---")
    st.subheader("Model Performance Comparison")
    fake_metric_col1, fake_metric_col2 = st.columns(2)
    fake_model_names = list(fake_metrics.keys())

    with fake_metric_col1:
        st.markdown(f"**{fake_model_names[0]}**")
        st.write("Accuracy:", f"{fake_metrics[fake_model_names[0]]['accuracy']:.3f}")
        st.write("Fake F1:", f"{fake_metrics[fake_model_names[0]]['fake_f1']:.3f}")

    with fake_metric_col2:
        st.markdown(f"**{fake_model_names[1]}**")
        st.write("Accuracy:", f"{fake_metrics[fake_model_names[1]]['accuracy']:.3f}")
        st.write("Fake F1:", f"{fake_metrics[fake_model_names[1]]['fake_f1']:.3f}")

    st.info(f"Best fake-news model: {fake_best_model_name}")
