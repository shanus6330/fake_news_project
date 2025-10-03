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
        df = pd.read_csv("archive (1)/data.csv")  # update path if needed
    except:
        st.error("❌ Could not find data.csv. Make sure it's uploaded in your repo.")
        return None
    return df

# ----------------------------
# Train Model
# ----------------------------
@st.cache_resource
def train_model(df):
    X = df["Body"].fillna("")  # use Body column (could also try Headline)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=200)),
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
