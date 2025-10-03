# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
data = pd.read_csv(r"C:\Users\MySurface\Downloads\fake_news_project\archive (1)\data.csv")

# 2. Combine Headline + Body into a single text column
data["content"] = data["Headline"].astype(str) + " " + data["Body"].astype(str)

X = data["content"]
y = data["Label"]   # label column (0 = Fake, 1 = Real)

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Convert text → TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 7. Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# 8. Quick test
sample_text = "Breaking news! Scientists discovered a new planet."
sample_tfidf = vectorizer.transform([sample_text])
print("Prediction:", "True News" if model.predict(sample_tfidf)[0] == 1 else "Fake News")
