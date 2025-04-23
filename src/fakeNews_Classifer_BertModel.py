
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
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class NewsDataset(Dataset):
    """Custom Dataset for loading news data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
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
        self.linear = nn.Linear(768, 2)  # 768 is BERT's hidden size
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
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
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
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

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['full_text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = BertNewsClassifier().to(device)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, true_labels = evaluate_model(model, val_loader, device)
    
    # Print metrics
    print("\nValidation Results:")
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(f"Precision: {precision_score(true_labels, predictions):.4f}")
    print(f"Recall: {recall_score(true_labels, predictions):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predictions):.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    main()