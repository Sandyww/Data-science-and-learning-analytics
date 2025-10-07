"""
Text Classification with Machine Learning

This script demonstrates:
- Feature extraction (Count Vectorizer, TF-IDF)
- Text classification using various algorithms
- Model evaluation and comparison
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    """Preprocess text for classification"""
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    
    return ' '.join(words)


def create_sample_dataset():
    """Create a sample text classification dataset"""
    print("=" * 50)
    print("CREATING SAMPLE DATASET")
    print("=" * 50)
    
    # Sample movie reviews dataset
    data = {
        'text': [
            "This movie is absolutely fantastic! Best film I've seen this year.",
            "Terrible movie, waste of time and money. Very disappointed.",
            "Great acting and amazing story. Highly recommend!",
            "Boring and predictable. Not worth watching.",
            "Excellent cinematography and wonderful performances.",
            "Poor script and bad acting. One of the worst movies ever.",
            "Loved every minute of it! A masterpiece.",
            "Awful experience. Would not recommend to anyone.",
            "Outstanding film with brilliant direction.",
            "Disappointing and dull. Expected much better.",
            "Incredible movie! Must watch for everyone.",
            "Waste of time. Terrible plot and poor execution.",
            "Beautiful story and amazing visuals. Loved it!",
            "Not good at all. Very boring and slow.",
            "Perfect movie! Everything about it was great.",
            "Horrible film. Couldn't even finish watching it.",
            "Superb acting and engaging storyline.",
            "Very bad movie. Don't waste your time.",
            "Fantastic performances by all actors. Highly entertaining!",
            "Poor quality and terrible direction. Avoid this movie."
        ],
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative'
        ]
    }
    
    df = pd.DataFrame(data)
    
    print(f"\nCreated dataset with {len(df)} samples")
    print(f"\nClass distribution:")
    print(df['sentiment'].value_counts())
    
    print("\nSample records:")
    print(df.head())
    
    return df


def feature_extraction_demo(df):
    """Demonstrate feature extraction techniques"""
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION")
    print("=" * 50)
    
    # Preprocess texts
    print("\n1. Preprocessing texts...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Sample preprocessed text:")
    print(df[['text', 'cleaned_text']].head(3))
    
    # Count Vectorizer
    print("\n2. Count Vectorizer (Bag of Words):")
    count_vectorizer = CountVectorizer(max_features=50)
    count_features = count_vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"Feature matrix shape: {count_features.shape}")
    print(f"Vocabulary size: {len(count_vectorizer.vocabulary_)}")
    print(f"Sample features: {list(count_vectorizer.vocabulary_.keys())[:10]}")
    
    # TF-IDF Vectorizer
    print("\n3. TF-IDF Vectorizer:")
    tfidf_vectorizer = TfidfVectorizer(max_features=50)
    tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"Feature matrix shape: {tfidf_features.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Sample features: {list(tfidf_vectorizer.vocabulary_.keys())[:10]}")
    
    return df, tfidf_vectorizer


def train_classification_models(df, vectorizer):
    """Train and evaluate multiple classification models"""
    print("\n" + "=" * 50)
    print("TRAINING CLASSIFICATION MODELS")
    print("=" * 50)
    
    # Prepare data
    X = vectorizer.transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Models to train
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    # Compare models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    results_df = results_df.sort_values('Accuracy', ascending=False)
    print(results_df)
    
    return models['Logistic Regression']


def predict_new_samples(model, vectorizer):
    """Demonstrate prediction on new samples"""
    print("\n" + "=" * 50)
    print("PREDICTING NEW SAMPLES")
    print("=" * 50)
    
    new_reviews = [
        "This is an amazing movie with great acting!",
        "Terrible film, complete waste of time.",
        "One of the best movies I have ever seen.",
        "Very disappointing and boring.",
        "Incredible storyline and fantastic performances!"
    ]
    
    print("\nPredicting sentiment for new reviews:")
    print("-" * 50)
    
    for review in new_reviews:
        # Preprocess
        cleaned_review = preprocess_text(review)
        
        # Vectorize
        features = vectorizer.transform([cleaned_review])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        print(f"\nReview: {review}")
        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence: Positive={probability[1]:.2f}, Negative={probability[0]:.2f}")


def ngram_features():
    """Demonstrate n-gram features"""
    print("\n" + "=" * 50)
    print("N-GRAM FEATURES")
    print("=" * 50)
    
    texts = [
        "machine learning is great",
        "deep learning and neural networks",
        "natural language processing"
    ]
    
    print("\nOriginal texts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    # Unigrams (1-gram)
    print("\n1. Unigrams (single words):")
    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
    unigram_features = unigram_vectorizer.fit_transform(texts)
    print(f"Features: {unigram_vectorizer.get_feature_names_out()}")
    
    # Bigrams (2-gram)
    print("\n2. Bigrams (word pairs):")
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
    bigram_features = bigram_vectorizer.fit_transform(texts)
    print(f"Features: {bigram_vectorizer.get_feature_names_out()}")
    
    # Combined (unigrams + bigrams)
    print("\n3. Combined (unigrams + bigrams):")
    combined_vectorizer = CountVectorizer(ngram_range=(1, 2))
    combined_features = combined_vectorizer.fit_transform(texts)
    print(f"Features: {combined_vectorizer.get_feature_names_out()}")


def main():
    """Main function to run all examples"""
    print("Welcome to Text Classification!\n")
    
    # Create dataset
    df = create_sample_dataset()
    
    # Feature extraction
    df, vectorizer = feature_extraction_demo(df)
    
    # Train models
    best_model = train_classification_models(df, vectorizer)
    
    # Predict new samples
    predict_new_samples(best_model, vectorizer)
    
    # N-gram features
    ngram_features()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
