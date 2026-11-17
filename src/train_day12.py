import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load local dataset
print("Loading dataset...")

df = pd.read_csv("data/amazon_reviews.csv")

print("Dataset loaded successfully!")
print("Total reviews:", len(df))
print(df.head())


# 2. Clean text
def clean_text(text):
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove numbers, punctuation and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning reviews...")

df["clean_content"] = df["content"].apply(clean_text)


# 3. Prepare features and labels
X = df["clean_content"]
y = df["label"]


# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))


# 5. TF-IDF
print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# 6. Train Logistic Regression
print("Training model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# 7. Evaluate model
print("Evaluating model...")

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    )
)


# 8. Most informative terms
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = model.coef_[0]

negative_indices = np.argsort(coefficients)[:15]
positive_indices = np.argsort(coefficients)[-15:][::-1]

negative_words = feature_names[negative_indices]
positive_words = feature_names[positive_indices]

print("\nMost informative NEGATIVE terms:")

for word in negative_words:
    print(word)


print("\nMost informative POSITIVE terms:")

for word in positive_words:
    print(word)


# 9. Test custom reviews
test_reviews = [
    "This product is amazing and works perfectly",
    "Terrible product and complete waste of money"
]

print("\nCustom Review Predictions:")

for review in test_reviews:

    cleaned_review = clean_text(review)

    review_vector = tfidf.transform([cleaned_review])

    prediction = model.predict(review_vector)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    print("\nReview:", review)
    print("Prediction:", sentiment)