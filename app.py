import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/shanus6330/fake_news_project/main/data.csv"
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        return None
    return df

# ----------------------------
# Train Model
# ----------------------------
@st.cache_resource
def train_model(df):
    # Combine Headline + Body
    X = (df['Headline'].fillna('') + " " + df['Body'].fillna(''))
    y = df['Label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression with class_weight='balanced'
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=200, class_weight='balanced')),
    ])

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    return model, acc

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")

df = load_data()
if df is not None:
    st.write("### Sample Data")
    st.write(df.head())

    model, acc = train_model(df)
    st.success(f"✅ Model trained with accuracy: {acc:.2f}")

    # User Input
    st.write("### Try it yourself")
    user_text = st.text_area("Enter a news headline or article text:")

    if st.button("Check if Fake or Real"):
        if user_text.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            pred = model.predict([user_text])[0]
            label = "🟢 Real News" if pred == 1 else "🔴 Fake News"
            st.subheader(f"Prediction: {label}")
else:
    st.error("Dataset could not be loaded. Please check the file path.")
