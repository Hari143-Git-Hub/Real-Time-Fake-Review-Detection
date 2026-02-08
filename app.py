from flask import Flask, render_template, request
import joblib
import re
import sqlite3
from datetime import datetime
from nltk.corpus import stopwords

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- DATABASE SETUP ----------------

def get_db():
    conn = sqlite3.connect("reviews.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT UNIQUE,
            result TEXT,
            risk TEXT,
            confidence TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# ---------------- HOME / ANALYZE PAGE ----------------

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    confidence = ""
    css_class = ""
    reasons = []
    analytics = {}
    risk_level = ""
    decision_summary = {}

    if request.method == 'POST':
        review = request.form['review']
        cleaned_review = clean_text(review)
        vector = vectorizer.transform([cleaned_review])

        # ML probabilities
        proba = model.predict_proba(vector)[0]
        classes = list(model.classes_)

        fake_prob = proba[classes.index("fake")] * 100
        genuine_prob = proba[classes.index("genuine")] * 100

        # Rule-based spam detection
        spam_words = ["buy now", "cheap", "best", "offer", "discount", "free", "limited"]
        review_lower = review.lower()
        strong_spam = False

        for word in spam_words:
            if word in review_lower:
                strong_spam = True
                reasons.append(f"Promotional keyword detected: '{word}'")

        # Final hybrid decision
        if fake_prob > genuine_prob or strong_spam:
            prediction = "fake"
            probability = fake_prob
        else:
            prediction = "genuine"
            probability = genuine_prob

        # Confidence interpretation
        if probability > 80:
            confidence_level = "High"
        elif probability > 60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        confidence = f"{confidence_level} ({probability:.2f}%)"

        # Risk Meter
        if fake_prob > 70:
            risk_level = "HIGH"
        elif fake_prob > 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Decision Summary
        decision_summary = {
            "Classification": "Fake" if prediction == "fake" else "Genuine",
            "Primary Reason": "Promotional language detected" if strong_spam else "No spam patterns found",
            "Risk Level": risk_level
        }

        # Analytics
        word_count = len(review.split())
        analytics["Word Count"] = word_count
        analytics["Spam Indicators"] = len(reasons)

        if prediction == "fake":
            result = "❌ Fake Review"
            css_class = "fake"
        else:
            result = "✅ Genuine Review"
            css_class = "genuine"

        # Explainability
        if not reasons:
            reasons.append("No promotional or spam language detected")
        if word_count < 5:
            reasons.append("Very short review – limited opinion expressed")
        else:
            reasons.append("Review length appears normal")

        # -------- INSERT OR UPDATE (BACKGROUND HISTORY SAVE) --------
        conn = get_db()
        conn.execute("""
            INSERT INTO reviews (review, result, risk, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(review) DO UPDATE SET
                result=excluded.result,
                risk=excluded.risk,
                confidence=excluded.confidence,
                created_at=excluded.created_at
        """, (
            review,
            prediction,
            risk_level,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        conn.commit()
        conn.close()

        # ❌ NO REDIRECT — SAME PAGE SHOW PHASES

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        css_class=css_class,
        reasons=reasons,
        analytics=analytics,
        risk_level=risk_level,
        decision_summary=decision_summary
    )

# ---------------- HISTORY PAGE ----------------

@app.route('/history')
def history():
    conn = get_db()
    reviews = conn.execute(
        "SELECT * FROM reviews ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("history.html", reviews=reviews)

# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=False)
