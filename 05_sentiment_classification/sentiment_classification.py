"""
Sentiment Classification using Logistic Regression

This script demonstrates:
- Building sentiment classifier with logistic regression
- Hyperparameter tuning
- Model evaluation and interpretation
- ROC curves and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, roc_auc_score)
from scipy.sparse import hstack
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_text(text):
    """Preprocess text for analysis"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    
    return ' '.join(words)


def create_comprehensive_dataset():
    """Create a comprehensive sentiment dataset"""
    print("=" * 50)
    print("CREATING COMPREHENSIVE DATASET")
    print("=" * 50)
    
    data = {
        'text': [
            "Absolutely love this! Best purchase ever!",
            "Terrible quality. Very disappointed.",
            "Great product. Works perfectly as described.",
            "Waste of money. Broke after first use.",
            "Excellent! Highly recommend.",
            "Poor quality. Not worth the price.",
            "Amazing product! Exceeded expectations.",
            "Horrible. Would not buy again.",
            "Perfect! Just what I needed.",
            "Bad experience. Customer service was terrible.",
            "Outstanding quality and great value!",
            "Awful product. Complete waste.",
            "Fantastic! Love everything about it.",
            "Very disappointing. Poor construction.",
            "Wonderful! Works like a charm.",
            "Not good. Stopped working quickly.",
            "Superb quality and fast shipping!",
            "Terrible. Regret this purchase.",
            "Brilliant product! Very satisfied.",
            "Poor quality. Do not recommend.",
            "Love it! Best in its category.",
            "Bad quality. Not as advertised.",
            "Incredible! Worth every penny.",
            "Disappointing. Expected better quality.",
            "Perfect product! Five stars!",
            "Horrible quality. Returned immediately.",
            "Great value! Very happy with purchase.",
            "Poor design. Breaks easily.",
            "Excellent quality! Highly satisfied.",
            "Terrible product. Avoid at all costs."
        ],
        'sentiment': [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0
        ]
    }
    
    df = pd.DataFrame(data)
    
    print(f"\nDataset size: {len(df)}")
    print(f"\nClass distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        label = 'Positive' if sentiment == 1 else 'Negative'
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def extract_all_features(df):
    """Extract text and custom features"""
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION")
    print("=" * 50)
    
    # Preprocess
    print("\n1. Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # TF-IDF features
    print("\n2. Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50)
    tfidf_features = tfidf.fit_transform(df['cleaned_text'])
    print(f"   TF-IDF shape: {tfidf_features.shape}")
    
    # Custom features
    print("\n3. Extracting custom features...")
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    df['punctuation_count'] = df['text'].apply(
        lambda x: sum([1 for char in x if char in string.punctuation])
    )
    
    custom_features = df[['text_length', 'word_count', 'exclamation_count', 'punctuation_count']].values
    
    # Scale custom features
    scaler = StandardScaler()
    custom_features_scaled = scaler.fit_transform(custom_features)
    
    # Combine features
    X = hstack([tfidf_features, custom_features_scaled])
    print(f"   Combined features shape: {X.shape}")
    
    return X, df['sentiment'], tfidf, scaler


def train_baseline_model(X, y):
    """Train baseline logistic regression model"""
    print("\n" + "=" * 50)
    print("BASELINE LOGISTIC REGRESSION MODEL")
    print("=" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining logistic regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba


def cross_validation_evaluation(X, y):
    """Perform cross-validation"""
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION")
    print("=" * 50)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")


def hyperparameter_tuning(X, y):
    """Tune hyperparameters using GridSearchCV"""
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    print("\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Grid search
    print("\nPerforming grid search...")
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    print("\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def visualize_results(y_test, y_pred, y_pred_proba):
    """Visualize model results"""
    print("\n" + "=" * 50)
    print("VISUALIZING RESULTS")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'sentiment_classification_results.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def predict_new_samples(model, tfidf, scaler):
    """Predict sentiment for new samples"""
    print("\n" + "=" * 50)
    print("PREDICTING NEW SAMPLES")
    print("=" * 50)
    
    new_texts = [
        "This is absolutely amazing! Love it!",
        "Terrible product. Complete waste of money.",
        "Good quality and fast delivery.",
        "Not satisfied. Poor quality.",
        "Excellent! Highly recommend this product!"
    ]
    
    print("\nPredicting sentiment for new texts:")
    print("-" * 70)
    
    for text in new_texts:
        # Preprocess
        cleaned = preprocess_text(text)
        
        # Extract features
        tfidf_feat = tfidf.transform([cleaned])
        
        custom_feat = np.array([[
            len(text),
            len(text.split()),
            text.count('!'),
            sum([1 for char in text if char in string.punctuation])
        ]])
        custom_feat_scaled = scaler.transform(custom_feat)
        
        # Combine
        X_new = hstack([tfidf_feat, custom_feat_scaled])
        
        # Predict
        prediction = model.predict(X_new)[0]
        probability = model.predict_proba(X_new)[0]
        
        sentiment = 'POSITIVE' if prediction == 1 else 'NEGATIVE'
        confidence = probability[prediction]
        
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: Negative={probability[0]:.4f}, Positive={probability[1]:.4f}")


def main():
    """Main function to run all examples"""
    print("Welcome to Sentiment Classification with Logistic Regression!\n")
    
    # Create dataset
    df = create_comprehensive_dataset()
    
    # Extract features
    X, y, tfidf, scaler = extract_all_features(df)
    
    # Train baseline model
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_baseline_model(X, y)
    
    # Cross-validation
    cross_validation_evaluation(X, y)
    
    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X, y)
    
    # Visualize results
    visualize_results(y_test, y_pred, y_pred_proba)
    
    # Predict new samples
    predict_new_samples(best_model, tfidf, scaler)
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
