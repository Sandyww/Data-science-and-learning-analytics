"""
Text Preprocessing

This script demonstrates:
- Text cleaning
- Tokenization
- Stop word removal
- Stemming and lemmatization
"""

import re
import string
import pandas as pd
import nltk
from collections import Counter

# Download required NLTK data (will download on first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def basic_text_cleaning():
    """Demonstrate basic text cleaning operations"""
    print("=" * 50)
    print("BASIC TEXT CLEANING")
    print("=" * 50)
    
    # Sample texts
    texts = [
        "Hello World! This is an EXAMPLE text with 123 numbers.",
        "Text preprocessing is IMPORTANT for NLP tasks!!!",
        "We need to remove @mentions, #hashtags, and URLs http://example.com",
        "Special characters like $, %, & should be handled carefully."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Original: {text}")
        
        # Lowercase
        text_lower = text.lower()
        print(f"   Lowercase: {text_lower}")
        
        # Remove punctuation
        text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
        print(f"   No punctuation: {text_no_punct}")
        
        # Remove numbers
        text_no_numbers = re.sub(r'\d+', '', text)
        print(f"   No numbers: {text_no_numbers}")
        
        # Remove extra whitespace
        text_clean = ' '.join(text.split())
        print(f"   Clean whitespace: {text_clean}")


def tokenization_demo():
    """Demonstrate tokenization techniques"""
    print("\n" + "=" * 50)
    print("TOKENIZATION")
    print("=" * 50)
    
    text = """
    Natural Language Processing (NLP) is a field of AI. 
    It helps computers understand human language. 
    Tokenization is the first step in text processing!
    """
    
    print(f"\nOriginal text: {text}")
    
    # Sentence tokenization
    print("\n1. Sentence Tokenization:")
    sentences = sent_tokenize(text)
    for i, sent in enumerate(sentences, 1):
        print(f"   Sentence {i}: {sent.strip()}")
    
    # Word tokenization
    print("\n2. Word Tokenization:")
    words = word_tokenize(text)
    print(f"   Words: {words}")
    print(f"   Total words: {len(words)}")
    
    # Simple split tokenization
    print("\n3. Simple Split Tokenization:")
    simple_words = text.split()
    print(f"   Words: {simple_words[:10]}...")  # Show first 10
    print(f"   Total words: {len(simple_words)}")


def stopword_removal():
    """Demonstrate stop word removal"""
    print("\n" + "=" * 50)
    print("STOP WORD REMOVAL")
    print("=" * 50)
    
    text = "This is an example sentence demonstrating the removal of stop words from text"
    
    print(f"\nOriginal text: {text}")
    
    # Get English stop words
    stop_words = set(stopwords.words('english'))
    print(f"\nNumber of English stop words: {len(stop_words)}")
    print(f"Sample stop words: {list(stop_words)[:10]}")
    
    # Tokenize
    words = word_tokenize(text.lower())
    print(f"\nTokenized: {words}")
    
    # Remove stop words
    filtered_words = [word for word in words if word not in stop_words]
    print(f"After removing stop words: {filtered_words}")
    print(f"\nOriginal word count: {len(words)}")
    print(f"Filtered word count: {len(filtered_words)}")


def stemming_demo():
    """Demonstrate stemming"""
    print("\n" + "=" * 50)
    print("STEMMING")
    print("=" * 50)
    
    stemmer = PorterStemmer()
    
    words = [
        'running', 'runs', 'ran', 'runner',
        'easily', 'easy', 'easier',
        'fairly', 'fairness', 'fair',
        'connection', 'connected', 'connecting'
    ]
    
    print("\nStemming examples (Porter Stemmer):")
    print(f"{'Original':<15} -> {'Stem':<15}")
    print("-" * 35)
    
    for word in words:
        stemmed = stemmer.stem(word)
        print(f"{word:<15} -> {stemmed:<15}")


def lemmatization_demo():
    """Demonstrate lemmatization"""
    print("\n" + "=" * 50)
    print("LEMMATIZATION")
    print("=" * 50)
    
    lemmatizer = WordNetLemmatizer()
    
    words = [
        'running', 'runs', 'ran', 'runner',
        'better', 'best', 'good',
        'am', 'is', 'are', 'was', 'were',
        'mice', 'mouse', 'children', 'child'
    ]
    
    print("\nLemmatization examples:")
    print(f"{'Original':<15} -> {'Lemma':<15}")
    print("-" * 35)
    
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')  # 'v' for verb
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, pos='n')  # 'n' for noun
        print(f"{word:<15} -> {lemma:<15}")


def complete_preprocessing_pipeline():
    """Demonstrate a complete preprocessing pipeline"""
    print("\n" + "=" * 50)
    print("COMPLETE PREPROCESSING PIPELINE")
    print("=" * 50)
    
    text = """
    Hello! This is an EXAMPLE of text preprocessing in Python. 
    We're going to clean this text, tokenize it, remove stop words, 
    and lemmatize the words. URLs like http://example.com should be removed!
    #NLP #TextProcessing @user123
    """
    
    print(f"\nOriginal text: {text}")
    
    # Step 1: Lowercase
    text = text.lower()
    print(f"\n1. Lowercase: {text}")
    
    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    print(f"\n2. Remove URLs: {text}")
    
    # Step 3: Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    print(f"\n3. Remove mentions/hashtags: {text}")
    
    # Step 4: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    print(f"\n4. Remove punctuation: {text}")
    
    # Step 5: Tokenize
    words = word_tokenize(text)
    print(f"\n5. Tokenize: {words}")
    
    # Step 6: Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    print(f"\n6. Remove stop words: {words}")
    
    # Step 7: Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    print(f"\n7. Lemmatize: {words}")
    
    # Final result
    cleaned_text = ' '.join(words)
    print(f"\nFinal cleaned text: {cleaned_text}")


def preprocessing_function(text):
    """Reusable preprocessing function"""
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


def apply_to_dataset():
    """Apply preprocessing to a dataset"""
    print("\n" + "=" * 50)
    print("APPLYING PREPROCESSING TO DATASET")
    print("=" * 50)
    
    # Create sample dataset
    data = {
        'text': [
            "I love this product! It's amazing!!!",
            "This is the worst experience ever.",
            "Not bad, but could be better. #review",
            "Excellent quality and fast shipping! http://example.com",
            "I'm very disappointed with this purchase."
        ],
        'label': ['positive', 'negative', 'neutral', 'positive', 'negative']
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal dataset:")
    print(df)
    
    # Apply preprocessing
    df['cleaned_text'] = df['text'].apply(preprocessing_function)
    
    print("\nDataset with cleaned text:")
    print(df)


def main():
    """Main function to run all examples"""
    print("Welcome to Text Preprocessing!\n")
    
    # Run all demonstrations
    basic_text_cleaning()
    tokenization_demo()
    stopword_removal()
    stemming_demo()
    lemmatization_demo()
    complete_preprocessing_pipeline()
    apply_to_dataset()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
