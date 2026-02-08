import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("reviews.csv")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

data['review'] = data['review'].apply(clean_text)

X = data['review']
y = data['label']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# -------------------------
# Model 1: Naive Bayes
# -------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# -------------------------
# Model 2: Logistic Regression
# -------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# -------------------------
# Select Best Model
# -------------------------
if lr_acc > nb_acc:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_acc = lr_acc
else:
    best_model = nb_model
    best_name = "Naive Bayes"
    best_acc = nb_acc

# Save best model & vectorizer
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"✅ Selected Model: {best_name}")
print(f"✅ Final Accuracy: {best_acc:.4f}")
print("✅ Best model and vectorizer saved successfully!")
