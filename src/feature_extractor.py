from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def create_features(cleaned_texts, max_features=3000):
    """
    Converts cleaned text into TF-IDF feature vectors.
    
    cleaned_texts: list/series of cleaned ticket text
    max_features: maximum number of words to consider as features
    
    Returns: TF-IDF matrix, fitted vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)   # considers single words AND two-word phrases
    )

    X = vectorizer.fit_transform(cleaned_texts)

    return X, vectorizer


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and labels into training and testing sets.
    
    test_size=0.2 means 20% of data is used for testing,
    80% is used for training.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   # ensures equal class proportion in train/test
    )

    return X_train, X_test, y_train, y_test