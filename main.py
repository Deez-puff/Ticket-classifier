# ============================================================
# Support Ticket Classification & Prioritization System
# ML Task 4 — Future Interns 2026
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from preprocessor import clean_text
from feature_extractor import create_features, split_data
from classifier import train_category_model, save_model, predict_priority, assign_priority_to_dataset
from evaluator import evaluate_model, plot_confusion_matrix

# ── 1. Load Dataset ───────────────────────────────────────
print("=" * 50)
print("   SUPPORT TICKET CLASSIFICATION SYSTEM")
print("=" * 50)

df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")
print(f"\n✅ Total tickets: {len(df)}")

# ── 2. Clean Ticket Text ──────────────────────────────────
print("\n⏳ Cleaning ticket text...")
df['cleaned'] = df['Document'].astype(str).apply(clean_text)
print("✅ Cleaning done!")

# ── 3. Create TF-IDF Features ─────────────────────────────
print("\n⏳ Creating TF-IDF features...")
X, vectorizer = create_features(df['cleaned'])
print(f"✅ Feature matrix shape: {X.shape}")

# ── 4. Split Data for Category Classification ─────────────
print("\n⏳ Splitting data for Category classification...")
y_category = df['Topic_group']
X_train, X_test, y_train, y_test = split_data(X, y_category)
print(f"✅ Training set: {X_train.shape[0]} | Testing set: {X_test.shape[0]}")

# ── 5. Train Category Classifier ──────────────────────────
print("\n⏳ Training Category Classification Model...")
category_model = train_category_model(X_train, y_train)
print("✅ Model trained!")

# ── 6. Quick Accuracy Check ───────────────────────────────
train_accuracy = category_model.score(X_train, y_train)
test_accuracy  = category_model.score(X_test, y_test)

print(f"\n── Quick Results ──")
print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Testing Accuracy:  {test_accuracy:.2%}")

# ── 7. Save Model ─────────────────────────────────────────
save_model(category_model, vectorizer,
           "data/category_model.pkl", "data/category_vectorizer.pkl")
print("\n✅ Model saved to data/ folder!")

# ── 8. Sample Category Predictions ────────────────────────
samples = [
    "My laptop screen is broken and won't turn on",
    "I need access to the shared HR drive",
    "The storage server mailbox is almost full"
]

print(f"\n── Sample Predictions ──")
for ticket in samples:
    cleaned_sample = clean_text(ticket)
    sample_vector = vectorizer.transform([cleaned_sample])
    prediction = category_model.predict(sample_vector)
    print(f"'{ticket}' -> {prediction[0]}")

# ── 9. Assign Priority to All Tickets ─────────────────────
print("\n⏳ Assigning priority levels...")
df['priority'] = assign_priority_to_dataset(df, 'cleaned')
print("✅ Priority assignment done!")

print(f"\n── Priority Distribution ──")
print(df['priority'].value_counts())

# ── 10. Sample Priority Predictions ───────────────────────
print(f"\n── Sample Priority Predictions ──")
priority_samples = [
    "The server is completely down, this is urgent!",
    "I'm having a minor issue with my email",
    "I would like to request access to the shared drive"
]

for ticket in priority_samples:
    cleaned_sample = clean_text(ticket)
    priority = predict_priority(cleaned_sample)
    print(f"'{ticket}' -> Priority: {priority}")

# ── 11. Evaluate Model ─────────────────────────────────────
print("\n⏳ Evaluating model performance...")
results = evaluate_model(category_model, X_test, y_test)

print(f"\n── Evaluation Metrics ──")
print(f"Accuracy:  {results['accuracy']:.2%}")
print(f"Precision: {results['precision']:.2%}")
print(f"Recall:    {results['recall']:.2%}")
print(f"F1-Score:  {results['f1_score']:.2%}")

print(f"\n── Full Classification Report ──")
print(results['report'])

# ── 12. Confusion Matrix ──────────────────────────────────
print("\n⏳ Generating confusion matrix...")
labels = sorted(df['Topic_group'].unique())
plot_confusion_matrix(y_test, results['y_pred'], labels, save_path="data/confusion_matrix.png")
plt.show()
print("✅ Confusion matrix saved to data/confusion_matrix.png!")

print("\n🎉 Support Ticket Classification Complete!")
print("=" * 50)