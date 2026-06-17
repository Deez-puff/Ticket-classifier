import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('src')

from preprocessor import clean_text
from feature_extractor import create_features, split_data
from classifier import train_category_model, save_model, load_model, predict_priority
from evaluator import evaluate_model, plot_confusion_matrix

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="wide")

st.title("🎫 Support Ticket Classification & Prioritization")
st.markdown("Upload support tickets, train a model, and classify new tickets by category and priority.")
st.divider()

# ── Sidebar — Mode Selection ──────────────────────────────
mode = st.sidebar.radio("Choose Mode:", ["Train New Model", "Classify a Ticket"])

if mode == "Train New Model":
    st.header("Step 1 — Upload Ticket Dataset")
    uploaded_file = st.file_uploader("Upload CSV file with ticket text and category", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded — {len(df)} tickets found!")

        with st.expander("👀 Preview Dataset"):
            st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            text_column = st.selectbox("Which column has ticket text?", df.columns.tolist())
        with col2:
            category_column = st.selectbox("Which column has the category label?", df.columns.tolist())

        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting...")

            progress.progress(20, text="⏳ Cleaning text...")
            df['cleaned'] = df[text_column].astype(str).apply(clean_text)

            progress.progress(40, text="⏳ Creating features...")
            X, vectorizer = create_features(df['cleaned'])

            progress.progress(60, text="⏳ Splitting data...")
            y = df[category_column]
            X_train, X_test, y_train, y_test = split_data(X, y)

            progress.progress(80, text="⏳ Training model...")
            model = train_category_model(X_train, y_train)

            progress.progress(95, text="⏳ Evaluating...")
            results = evaluate_model(model, X_test, y_test)

            save_model(model, vectorizer, "data/category_model.pkl", "data/category_vectorizer.pkl")
            progress.progress(100, text="✅ Done!")

            st.divider()
            st.header("📊 Model Performance")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{results['accuracy']:.2%}")
            m2.metric("Precision", f"{results['precision']:.2%}")
            m3.metric("Recall", f"{results['recall']:.2%}")
            m4.metric("F1-Score", f"{results['f1_score']:.2%}")

            with st.expander("📋 Full Classification Report"):
                st.text(results['report'])

            st.subheader("🗺️ Confusion Matrix")
            labels = sorted(y.unique())
            fig = plot_confusion_matrix(y_test, results['y_pred'], labels)
            st.pyplot(fig)

            st.success("✅ Model trained and saved! Switch to 'Classify a Ticket' mode to use it.")

else:
    st.header("Classify a New Support Ticket")

    ticket_text = st.text_area(
        "Enter the support ticket text:",
        height=150,
        placeholder="e.g. My laptop screen is broken and won't turn on, this is urgent!"
    )

    if st.button("🔍 Classify Ticket", type="primary", use_container_width=True):
        if not ticket_text.strip():
            st.error("❌ Please enter ticket text first!")
        else:
            try:
                model, vectorizer = load_model("data/category_model.pkl", "data/category_vectorizer.pkl")

                cleaned = clean_text(ticket_text)
                vector = vectorizer.transform([cleaned])
                category = model.predict(vector)[0]
                priority = predict_priority(cleaned)

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("📂 Predicted Category", category)
                with c2:
                    priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.metric("⚡ Priority Level", f"{priority_color.get(priority,'')} {priority}")

                # Show confidence scores
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vector)[0]
                    prob_df = pd.DataFrame({
                        "Category": model.classes_,
                        "Confidence": probs
                    }).sort_values("Confidence", ascending=False)

                    st.subheader("📊 Confidence Breakdown")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(prob_df['Category'], prob_df['Confidence'], color='#3498db')
                    ax.set_xlabel("Confidence")
                    ax.invert_yaxis()
                    st.pyplot(fig)

            except FileNotFoundError:
                st.error("❌ No trained model found! Please train a model first in 'Train New Model' mode.")