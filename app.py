# app.py

import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")
st.write("Check whether a news article is Real or Fake using Machine Learning.")

# Input text
headline = st.text_input("Enter Headline")
body = st.text_area("Enter News Body")

if st.button("Check News"):
    if headline.strip() or body.strip():
        # Combine inputs
        content = headline + " " + body
        input_tfidf = vectorizer.transform([content])
        prediction = model.predict(input_tfidf)[0]

        if prediction == 1:
            st.success("✅ This looks like Real News")
        else:
            st.error("❌ This looks like Fake News")
    else:
        st.warning("Please enter at least a headline or body.")
