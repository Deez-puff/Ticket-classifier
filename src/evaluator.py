from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(model, X_test, y_test, labels=None):
    """
    Evaluates a trained classification model and returns
    accuracy, precision, recall, f1-score and a full report.
    """
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "report": report,
        "y_pred": y_pred
    }


def plot_confusion_matrix(y_test, y_pred, labels, save_path=None):
    """
    Plots and optionally saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix — Category Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return plt.gcf()