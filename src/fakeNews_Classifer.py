"""
Fake News Detection using Naive Bayes

This script implements a Naive Bayes classifier for detecting fake news articles.
It uses a similar approach to spam detection but is specifically adapted for news article classification.

The script:
1. Loads and preprocesses news article data
2. Implements a Naive Bayes classifier
3. Trains the model on real/fake news datasets
4. Evaluates performance
5. Makes predictions on new data
"""

# Import required libraries
import pandas as pd
import numpy as np
import re
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import zipfile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """Handles all text preprocessing tasks for the news articles"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """
        Preprocess the text by:
        1. Converting to lowercase
        2. Removing special characters and numbers
        3. Tokenizing
        4. Removing stopwords
        5. Removing short words (length < 3)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens

class NaiveBayesClassifier:
    """
    Naive Bayes classifier implementation for text classification.
    Uses log probabilities to prevent numerical underflow.
    """
    
    def __init__(self):
        self.vocab = set()
        self.word_counts = {'real': Counter(), 'fake': Counter()}
        self.class_counts = {'real': 0, 'fake': 0}
        self.total_docs = 0
        self.class_word_counts = {'real': 0, 'fake': 0}
    
    def train(self, texts, labels):
        """Train the classifier on the given texts and labels"""
        self.total_docs = len(texts)
        
        # Count documents per class and build vocabulary
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            
            # Update word counts and vocabulary
            for word in text:
                self.vocab.add(word)
                self.word_counts[label][word] += 1
                self.class_word_counts[label] += 1
    
    def get_word_prob(self, word, label):
        """Calculate P(word|label) with Laplace smoothing"""
        count = self.word_counts[label].get(word, 0)
        return (count + 1) / (self.class_word_counts[label] + len(self.vocab))
    
    def predict(self, text):
        """Predict the class of the given text"""
        scores = {}
        
        for label in ['real', 'fake']:
            # Prior probability (in log space)
            scores[label] = math.log(self.class_counts[label] / self.total_docs)
            
            # Add log probabilities for each word
            for word in text:
                if word in self.vocab:
                    prob = self.get_word_prob(word, label)
                    scores[label] += math.log(prob)
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def predict_proba(self, text):
        """Return probability scores for both classes"""
        scores = {}
        
        for label in ['real', 'fake']:
            scores[label] = math.log(self.class_counts[label] / self.total_docs)
            
            for word in text:
                if word in self.vocab:
                    prob = self.get_word_prob(word, label)
                    scores[label] += math.log(prob)
        
        # Convert log probabilities to regular probabilities
        max_score = max(scores.values())
        scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(scores.values())
        return {k: v/total for k, v in scores.items()}

def load_data():
    """Load and combine the true and fake news datasets"""
    # Load datasets
    true_df = pd.read_csv('../datasets/True.csv')
    fake_df = pd.read_csv('../datasets/Fake.csv')
    
    # Add labels
    true_df['label'] = 'real'
    fake_df['label'] = 'fake'
    
    # Combine datasets
    df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Combine title and text for better classification
    df['full_text'] = df['title'] + ' ' + df['text']
    
    return df

def load_test_data(preprocessor):
    """Load and preprocess the test dataset"""
    with zipfile.ZipFile('../datasets/wellsFakeNewsClass.zip', 'r') as zip_ref:
        # Extract the CSV file name from the zip
        csv_file = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
        
        # Read the CSV file
        with zip_ref.open(csv_file) as f:
            test_df = pd.read_csv(f)
    
    # Preprocess the test data
    test_df['tokens'] = test_df['text'].apply(preprocessor.preprocess)
    
    return test_df

def get_most_indicative_words(classifier, n=20):
    """Find the most indicative words for each class"""
    word_scores = {'real': [], 'fake': []}
    
    for word in classifier.vocab:
        # Calculate the ratio of probabilities
        p_real = classifier.get_word_prob(word, 'real')
        p_fake = classifier.get_word_prob(word, 'fake')
        
        # Store word and score
        if p_real > p_fake:
            word_scores['real'].append((word, p_real/p_fake))
        else:
            word_scores['fake'].append((word, p_fake/p_real))
    
    # Sort and get top n words for each class
    for label in word_scores:
        word_scores[label] = sorted(word_scores[label], 
                                  key=lambda x: x[1], 
                                  reverse=True)[:n]
    
    return word_scores

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix using seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    preprocessor = TextPreprocessor()
    df['tokens'] = df['full_text'].apply(preprocessor.preprocess)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df['tokens'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # Train classifier
    print("Training classifier...")
    classifier = NaiveBayesClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate on validation set
    print("Evaluating classifier...")
    y_pred = [classifier.predict(text) for text in X_val]
    
    # Print metrics
    print("\nValidation Results:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, pos_label='fake'):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred, pos_label='fake'):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred, pos_label='fake'):.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, y_pred)
    
    # Get most indicative words
    print("\nAnalyzing most indicative words...")
    indicative_words = get_most_indicative_words(classifier)
    
    print("\nMost indicative words for real news:")
    for word, score in indicative_words['real']:
        print(f"{word}: {score:.2f}")
    
    print("\nMost indicative words for fake news:")
    for word, score in indicative_words['fake']:
        print(f"{word}: {score:.2f}")
    
    # Process test data
    print("\nProcessing test data...")
    test_df = load_test_data(preprocessor)
    
    # Make predictions
    print("Making predictions on test data...")
    test_df['predicted_label'] = [classifier.predict(text) for text in test_df['tokens']]
    test_df['prediction_confidence'] = [max(classifier.predict_proba(text).values()) 
                                      for text in test_df['tokens']]
    
    # Save predictions
    predictions_df = test_df[['id', 'predicted_label', 'prediction_confidence']]
    predictions_df.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_df['prediction_confidence'], bins=50)
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    main()