"""
Fake News Detection using BERT

This script implements a BERT-based classifier for detecting fake news articles.
BERT is particularly good at this task because it:
1. Understands context and word relationships
2. Has been pre-trained on a large corpus of text
3. Can capture complex patterns in language

The script:
1. Loads and preprocesses news article data
2. Fine-tunes a pre-trained BERT model
3. Evaluates performance
4. Makes predictions on new data
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix

class NewsDataset(Dataset):
    """Custom Dataset for loading news data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        # Convert texts and labels to lists to avoid pandas indexing issues
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.labels = labels.tolist() if isinstance(labels, pd.Series) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(1 if label == 'fake' else 0, dtype=torch.long)
        }

class BertNewsClassifier(nn.Module):
    """BERT-based classifier for fake news detection"""
    
    def __init__(self, dropout=0.5):
        super(BertNewsClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Classifier layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification layers
        x = self.dropout(pooled_output)
        x = self.linear(x)
        return self.softmax(x)

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
    
    # Combine title and text
    df['full_text'] = df['title'] + ' ' + df['text']
    
    return df

def train_model(model, train_loader, val_loader, device, epochs=3):
    """Train the BERT model"""
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """Evaluate the model's performance"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels

def setup_visuals_directory():
    """Create a directory for saving visualizations and results"""
    base_dir = 'visuals_BERT'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
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
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{phase.lower()}.png'))
    plt.close()

def evaluate_predictions(y_true, y_pred, save_dir, phase="Validation"):
    """Evaluate predictions and save results"""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
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
    for label in [0, 1]:
        class_name = 'Real' if label == 0 else 'Fake'
        class_precision = precision_score(y_true, y_pred, pos_label=label)
        class_recall = recall_score(y_true, y_pred, pos_label=label)
        class_f1 = f1_score(y_true, y_pred, pos_label=label)
        print(f"\n{class_name} class:")
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

def plot_training_history(train_losses, val_losses, save_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def load_test_data(tokenizer):
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
        
        return test_df
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None

def main():
    # Create directory for this run
    save_dir = setup_visuals_directory()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Split data into train, validation, and test sets
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['full_text'].values, df['label'].values,  # Convert to numpy arrays
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
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset = NewsDataset(X_val, y_val, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Initialize model
    model = BertNewsClassifier().to(device)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, save_dir)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions, val_true_labels = evaluate_model(model, val_loader, device)
    val_metrics = evaluate_predictions(val_true_labels, val_predictions, save_dir, "Validation")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions, test_true_labels = evaluate_model(model, test_loader, device)
    test_metrics = evaluate_predictions(test_true_labels, test_predictions, save_dir, "Test")
    
    # Process external test data if available
    print("\nProcessing external test data...")
    test_df = load_test_data(tokenizer)
    
    if test_df is not None:
        # Create dataset and loader for external test data
        external_test_dataset = NewsDataset(test_df['text'], test_df['label'], tokenizer)
        external_test_loader = DataLoader(external_test_dataset, batch_size=16)
        
        # Make predictions
        model.eval()
        external_predictions = []
        external_confidences = []
        
        with torch.no_grad():
            for batch in external_test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                external_predictions.extend(predicted.cpu().numpy())
                external_confidences.extend(confidence.cpu().numpy())
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'id': range(len(test_df)) if 'id' not in test_df.columns else test_df['id'],
            'predicted_label': ['fake' if p == 1 else 'real' for p in external_predictions],
            'prediction_confidence': external_confidences
        })
        
        predictions_df.to_csv(os.path.join(save_dir, 'external_test_predictions.csv'), index=False)
        
        # If external test data has true labels, evaluate performance
        if 'label' in test_df.columns:
            external_true_labels = [1 if label == 'fake' else 0 for label in test_df['label']]
            external_test_metrics = evaluate_predictions(
                external_true_labels,
                external_predictions,
                save_dir,
                "External_Test"
            )
        
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