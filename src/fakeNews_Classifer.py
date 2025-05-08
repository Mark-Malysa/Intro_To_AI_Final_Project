#!/usr/bin/env python3

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
import seaborn as sns
import matplotlib.pyplot as plt
import ssl
import nltk
import os
from datetime import datetime

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """Handles all text preprocessing tasks for the news articles"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Warning: Could not load stopwords, using a basic set")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def preprocess(self, text):
        """
        Preprocess the text by:
        1. Converting to lowercase
        2. Removing special characters and numbers
        3. Tokenizing
        4. Removing stopwords
        5. Removing short words (length < 3)
        """
        # Handle potential None or non-string inputs
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Simple word splitting as backup if NLTK tokenization fails
        # try:
        #     tokens = word_tokenize(text)
        # except:
        #     print("Warning: NLTK tokenization failed, falling back to simple split")
        #     tokens = text.split()
        
        tokens = text.split()

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
    try:
        with zipfile.ZipFile('../datasets/wellsFakeNewsClass.zip', 'r') as zip_ref:
            csv_file = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
            
            with zip_ref.open(csv_file) as f:
                test_df = pd.read_csv(f)
        
        print("Test dataset columns:", test_df.columns.tolist())
        
        # Convert numeric labels to string labels if needed
        if 'label' in test_df.columns:
            test_df['label'] = test_df['label'].map({0: 'real', 1: 'fake'})
        
        # Preprocess the test data
        test_df['tokens'] = test_df['text'].apply(preprocessor.preprocess)
        
        return test_df
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None

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

def setup_visuals_directory():
    """Create a directory for saving visualizations and results"""
    # Create base directory
    base_dir = 'visuals_NB'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(run_dir)
    
    return run_dir

def plot_confusion_matrix(y_true, y_pred, save_dir, phase="Validation"):
    """Plot confusion matrix using seaborn and save it"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {phase}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{phase.lower()}.png'))
    plt.close()

def plot_confidence_analysis(confidence_scores, predicted_labels, save_dir, title="Confidence Score Distribution"):
    """Plot confidence score distribution and save it"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Overall confidence distribution
    plt.subplot(1, 2, 1)
    plt.hist(confidence_scores, bins=50)
    plt.title('Overall Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    
    # Plot 2: Box plot of confidence by predicted class
    plt.subplot(1, 2, 2)
    sns.boxplot(x=predicted_labels, y=confidence_scores)
    plt.title('Confidence Scores by Predicted Class')
    plt.xlabel('Predicted Class')
    plt.ylabel('Confidence Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'))
    plt.close()

def evaluate_predictions(y_true, y_pred, save_dir, phase="Validation"):
    """Evaluate predictions and save results"""
    # Convert labels to consistent format
    label_map = {'real': 'real', 'fake': 'fake', 0: 'real', 1: 'fake'}
    y_true = [label_map.get(str(label), label) for label in y_true]
    y_pred = [label_map.get(str(label), label) for label in y_pred]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='fake')
    recall = recall_score(y_true, y_pred, pos_label='fake')
    f1 = f1_score(y_true, y_pred, pos_label='fake')
    
    # Print detailed metrics
    print(f"\n{phase} Set Metrics:")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate and print class-wise metrics
    print("\nClass-wise Performance:")
    print(f"{'='*50}")
    for label in ['real', 'fake']:
        class_precision = precision_score(y_true, y_pred, pos_label=label)
        class_recall = recall_score(y_true, y_pred, pos_label=label)
        class_f1 = f1_score(y_true, y_pred, pos_label=label)
        print(f"\n{label.upper()} class:")
        print(f"Precision: {class_precision:.4f}")
        print(f"Recall: {class_recall:.4f}")
        print(f"F1 Score: {class_f1:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'='*50}")
    print("Predicted:")
    print("          Real    Fake")
    print(f"Real    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"Fake    {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    # Save metrics to file
    metrics = {
        'phase': phase,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, f'metrics_{phase.lower()}.csv'), index=False)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_dir, phase)
    
    return metrics

def analyze_misclassifications(texts, true_labels, predicted_labels, confidence_scores):
    """
    Analyze misclassified examples
    """
    # Convert labels to consistent format
    label_map = {'real': 'real', 'fake': 'fake', 0: 'real', 1: 'fake'}
    true_labels = [label_map.get(str(label), label) for label in true_labels]
    predicted_labels = [label_map.get(str(label), label) for label in predicted_labels]
    
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) if true != pred]
    
    print("\nMisclassification Analysis:")
    print(f"{'='*50}")
    print(f"Total misclassified examples: {len(misclassified_indices)}")
    
    if misclassified_indices:
        # Analyze confidence distribution for misclassified examples
        misclassified_confidences = [confidence_scores[i] for i in misclassified_indices]
        
        plt.figure(figsize=(10, 6))
        plt.hist(misclassified_confidences, bins=20)
        plt.title('Confidence Distribution for Misclassified Examples')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.show()
        
        # Show some examples of high-confidence mistakes
        print("\nHigh Confidence Mistakes:")
        print(f"{'='*50}")
        sorted_mistakes = sorted(zip(misclassified_indices, misclassified_confidences), 
                               key=lambda x: x[1], reverse=True)
        
        for idx, conf in sorted_mistakes[:5]:  # Show top 5 mistakes
            print(f"\nConfidence: {conf:.4f}")
            print(f"True Label: {true_labels[idx]}")
            print(f"Predicted: {predicted_labels[idx]}")
            print(f"Text excerpt: {texts[idx][:200]}...")
            print("-" * 50)

def calculate_class_wise_metrics(y_true, y_pred):
    """
    Calculate class-wise performance metrics
    """
    # Convert labels to consistent format
    label_map = {'real': 'real', 'fake': 'fake', 0: 'real', 1: 'fake'}
    y_true = [label_map.get(str(label), label) for label in y_true]
    y_pred = [label_map.get(str(label), label) for label in y_pred]
    
    classes = ['real', 'fake']
    metrics = {}
    
    for cls in classes:
        metrics[cls] = {
            'precision': precision_score(y_true, y_pred, pos_label=cls),
            'recall': recall_score(y_true, y_pred, pos_label=cls),
            'f1': f1_score(y_true, y_pred, pos_label=cls)
        }
    
    print("\nClass-wise Performance Metrics:")
    print(f"{'='*50}")
    for cls in classes:
        print(f"\n{cls.upper()} class:")
        for metric, value in metrics[cls].items():
            print(f"{metric}: {value:.4f}")

def main():
    # Create directory for this run
    save_dir = setup_visuals_directory()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    preprocessor = TextPreprocessor()
    df['tokens'] = df['full_text'].apply(preprocessor.preprocess)
    
    # Split data into train, validation, and test sets
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['tokens'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Second split: 80% train, 20% validation (of the remaining 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"Data split sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train classifier
    print("Training classifier...")
    classifier = NaiveBayesClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate on validation set
    print("Evaluating classifier on validation set...")
    y_val_pred = [classifier.predict(text) for text in X_val]
    val_metrics = evaluate_predictions(y_val, y_val_pred, save_dir, "Validation")
    
    # Evaluate on test set
    print("Evaluating classifier on test set...")
    y_test_pred = [classifier.predict(text) for text in X_test]
    test_metrics = evaluate_predictions(y_test, y_test_pred, save_dir, "Test")
    
    # Get most indicative words
    print("Analyzing most indicative words...")
    indicative_words = get_most_indicative_words(classifier)
    
    # Save indicative words
    for label in ['real', 'fake']:
        words_df = pd.DataFrame(indicative_words[label], columns=['word', 'score'])
        words_df.to_csv(os.path.join(save_dir, f'indicative_words_{label}.csv'), index=False)
    
    # Process external test data if available
    print("Processing external test data...")
    test_df = load_test_data(preprocessor)
    
    if test_df is not None:
        # Make predictions
        test_df['predicted_label'] = [classifier.predict(text) for text in test_df['tokens']]
        test_df['prediction_confidence'] = [max(classifier.predict_proba(text).values()) 
                                          for text in test_df['tokens']]
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'id': range(len(test_df)) if 'id' not in test_df.columns else test_df['id'],
            'predicted_label': test_df['predicted_label'],
            'prediction_confidence': test_df['prediction_confidence']
        })
        
        predictions_df.to_csv(os.path.join(save_dir, 'external_test_predictions.csv'), index=False)
        
        # Plot confidence analysis
        plot_confidence_analysis(test_df['prediction_confidence'],
                               test_df['predicted_label'],
                               save_dir)
        
        # If external test data has true labels, evaluate performance
        if 'label' in test_df.columns:
            external_test_metrics = evaluate_predictions(test_df['label'], 
                                                      test_df['predicted_label'],
                                                      save_dir,
                                                      "External_Test")
        
        # Save summary statistics
        summary_stats = {
            'total_predictions': len(predictions_df),
            'avg_confidence': predictions_df['prediction_confidence'].mean(),
            'median_confidence': predictions_df['prediction_confidence'].median(),
            'prediction_distribution': predictions_df['predicted_label'].value_counts().to_dict()
        }
        
        pd.DataFrame([summary_stats]).to_csv(os.path.join(save_dir, 'summary_stats.csv'), index=False)
        
        print(f"\nAll results and visualizations saved to: {save_dir}")
    else:
        print("No external test data available")

if __name__ == "__main__":
    main()