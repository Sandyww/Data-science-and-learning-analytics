"""
Feature Engineering for Text Data

This script demonstrates:
- Text-based features (TF-IDF, n-grams)
- Custom features (length, punctuation count, etc.)
- Feature combination and selection
- Feature scaling
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
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
    """Preprocess text for analysis"""
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
    """Create a sample sentiment dataset"""
    print("=" * 50)
    print("CREATING SAMPLE DATASET")
    print("=" * 50)
    
    data = {
        'text': [
            "This product is absolutely amazing! Best purchase ever!!!",
            "Terrible quality. Broke after one day. Very disappointed.",
            "Great value for money. Works perfectly!",
            "Waste of money. Poor quality and bad customer service.",
            "Excellent product! Highly recommend to everyone.",
            "Don't buy this. Complete garbage.",
            "Love it! Exactly what I needed.",
            "Horrible experience. Would not recommend.",
            "Outstanding quality and fast delivery!",
            "Very bad product. Not worth it at all.",
            "Perfect! Could not be happier with this purchase.",
            "Awful product. Stopped working after a week.",
            "Fantastic! Exceeded all my expectations.",
            "Poor quality. Very disappointing purchase.",
            "Amazing product with great features!",
            "Terrible. Worst purchase I've ever made.",
            "Wonderful product! Works like a charm.",
            "Bad quality and overpriced.",
            "Superb! Best in its class.",
            "Not good at all. Complete waste of money."
        ],
        'sentiment': [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0
        ]
    }
    
    df = pd.DataFrame(data)
    
    print(f"\nCreated dataset with {len(df)} samples")
    print(f"\nClass distribution:")
    print(df['sentiment'].value_counts())
    
    print("\nSample records:")
    print(df.head())
    
    return df


def extract_text_features(df):
    """Extract TF-IDF features from text"""
    print("\n" + "=" * 50)
    print("EXTRACTING TEXT FEATURES (TF-IDF)")
    print("=" * 50)
    
    # Preprocess texts
    print("\n1. Preprocessing texts...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Extract TF-IDF features with unigrams and bigrams
    print("\n2. Extracting TF-IDF features (unigrams + bigrams):")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100,
        min_df=1,
        max_df=0.8
    )
    
    tfidf_features = tfidf.fit_transform(df['cleaned_text'])
    
    print(f"TF-IDF feature matrix shape: {tfidf_features.shape}")
    print(f"Number of features: {len(tfidf.vocabulary_)}")
    print(f"\nSample features: {list(tfidf.vocabulary_.keys())[:15]}")
    
    return df, tfidf, tfidf_features


def extract_custom_features(df):
    """Extract custom features from text"""
    print("\n" + "=" * 50)
    print("EXTRACTING CUSTOM FEATURES")
    print("=" * 50)
    
    # 1. Text length
    df['text_length'] = df['text'].apply(len)
    print("\n1. Text length:")
    print(df[['text', 'text_length']].head())
    
    # 2. Word count
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    print("\n2. Word count:")
    print(df[['text', 'word_count']].head())
    
    # 3. Average word length
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
    )
    print("\n3. Average word length:")
    print(df[['text', 'avg_word_length']].head())
    
    # 4. Punctuation count
    df['punctuation_count'] = df['text'].apply(
        lambda x: sum([1 for char in x if char in string.punctuation])
    )
    print("\n4. Punctuation count:")
    print(df[['text', 'punctuation_count']].head())
    
    # 5. Exclamation mark count
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    print("\n5. Exclamation mark count:")
    print(df[['text', 'exclamation_count']].head())
    
    # 6. Uppercase word count
    df['uppercase_count'] = df['text'].apply(
        lambda x: sum([1 for word in x.split() if word.isupper()])
    )
    print("\n6. Uppercase word count:")
    print(df[['text', 'uppercase_count']].head())
    
    # 7. Capital letter ratio
    df['capital_ratio'] = df['text'].apply(
        lambda x: sum([1 for char in x if char.isupper()]) / len(x) if len(x) > 0 else 0
    )
    print("\n7. Capital letter ratio:")
    print(df[['text', 'capital_ratio']].head())
    
    # Custom feature names
    custom_feature_names = [
        'text_length', 'word_count', 'avg_word_length', 
        'punctuation_count', 'exclamation_count', 
        'uppercase_count', 'capital_ratio'
    ]
    
    return df, custom_feature_names


def combine_features(df, tfidf_features, custom_feature_names):
    """Combine text features with custom features"""
    print("\n" + "=" * 50)
    print("COMBINING FEATURES")
    print("=" * 50)
    
    # Get custom features as numpy array
    custom_features = df[custom_feature_names].values
    
    print(f"\nTF-IDF features shape: {tfidf_features.shape}")
    print(f"Custom features shape: {custom_features.shape}")
    
    # Scale custom features
    scaler = StandardScaler()
    custom_features_scaled = scaler.fit_transform(custom_features)
    
    print(f"Scaled custom features shape: {custom_features_scaled.shape}")
    
    # Combine features
    combined_features = hstack([tfidf_features, custom_features_scaled])
    
    print(f"\nCombined features shape: {combined_features.shape}")
    print(f"Total number of features: {combined_features.shape[1]}")
    
    return combined_features, scaler


def feature_importance_analysis(df, custom_feature_names):
    """Analyze custom feature importance"""
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Group by sentiment and calculate mean
    sentiment_map = {0: 'Negative', 1: 'Positive'}
    
    print("\nAverage feature values by sentiment:")
    print("-" * 70)
    
    feature_stats = df.groupby('sentiment')[custom_feature_names].mean()
    feature_stats.index = feature_stats.index.map(sentiment_map)
    
    print(feature_stats)
    
    # Calculate difference
    print("\nFeature differences (Positive - Negative):")
    print("-" * 70)
    
    diff = feature_stats.loc['Positive'] - feature_stats.loc['Negative']
    diff_sorted = diff.sort_values(ascending=False)
    
    for feature, value in diff_sorted.items():
        direction = "higher" if value > 0 else "lower"
        print(f"{feature:.<30} {direction:>10} by {abs(value):.4f}")


def feature_engineering_pipeline():
    """Demonstrate complete feature engineering pipeline"""
    print("\n" + "=" * 50)
    print("COMPLETE FEATURE ENGINEERING PIPELINE")
    print("=" * 50)
    
    # Create dataset
    df = create_sample_dataset()
    
    # Extract text features
    df, tfidf, tfidf_features = extract_text_features(df)
    
    # Extract custom features
    df, custom_feature_names = extract_custom_features(df)
    
    # Combine features
    combined_features, scaler = combine_features(df, tfidf_features, custom_feature_names)
    
    # Analyze feature importance
    feature_importance_analysis(df, custom_feature_names)
    
    print("\n" + "=" * 50)
    print("Feature engineering completed!")
    print(f"Final feature matrix shape: {combined_features.shape}")
    print("=" * 50)
    
    return df, combined_features, tfidf, scaler, custom_feature_names


def main():
    """Main function to run all examples"""
    print("Welcome to Feature Engineering for Text Data!\n")
    
    # Run complete pipeline
    feature_engineering_pipeline()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
