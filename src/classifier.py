from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib

def train_category_model(X_train, y_train):
    """
    Trains a Logistic Regression model to classify
    tickets into categories (Ticket Type).
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_priority_model(X_train, y_train):
    """
    Trains a Logistic Regression model to predict
    ticket priority (High/Medium/Low/Critical).
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, vectorizer, model_path, vectorizer_path):
    """
    Saves trained model and vectorizer to disk
    so we don't have to retrain every time.
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load_model(model_path, vectorizer_path):
    """
    Loads a previously trained model and vectorizer from disk.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ── Rule-Based Priority System ────────────────────────────

HIGH_PRIORITY_WORDS = [
    "urgent", "asap", "immediately", "critical", "emergency",
    "down", "broken", "crash", "failure", "not working",
    "unable to access", "blocked", "outage", "severe"
]

MEDIUM_PRIORITY_WORDS = [
    "issue", "problem", "error", "bug", "trouble",
    "slow", "delay", "warning", "incorrect"
]

LOW_PRIORITY_WORDS = [
    "request", "inquiry", "question", "information",
    "please advise", "would like", "when possible", "minor"
]


def predict_priority(text):
    """
    Assigns priority (High/Medium/Low) based on
    urgency keywords found in the cleaned ticket text.
    """
    text = text.lower()

    high_score   = sum(1 for word in HIGH_PRIORITY_WORDS if word in text)
    medium_score = sum(1 for word in MEDIUM_PRIORITY_WORDS if word in text)
    low_score    = sum(1 for word in LOW_PRIORITY_WORDS if word in text)

    if high_score > 0:
        return "High"
    elif medium_score > 0:
        return "Medium"
    elif low_score > 0:
        return "Low"
    else:
        return "Medium"   # default fallback if no keywords matched


def assign_priority_to_dataset(df, text_column):
    """
    Applies priority prediction to an entire dataframe column.
    """
    return df[text_column].apply(predict_priority)