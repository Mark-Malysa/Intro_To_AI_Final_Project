# Intro_To_AI_Final_Project

## Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```

## Implementations

### 1. Naive Bayes Classifier (`fakeNews_Classifer.py`)
A traditional machine learning approach using Naive Bayes for text classification.

#### Features:
- Text preprocessing using NLTK
- TF-IDF vectorization
- Naive Bayes classification
- Comprehensive evaluation metrics
- Visualization of results

#### How to Run:
```bash
python src/fakeNews_Classifer.py
```

#### Output:
- Creates a `visuals_NB` directory with timestamped subdirectories
- Saves confusion matrices, metrics, and predictions
- Prints detailed evaluation metrics during execution

### 2. BERT-based Classifier (`fakeNews_Classifer_BertModel.py`)
A deep learning approach using the BERT (Bidirectional Encoder Representations from Transformers) model.

#### Features:
- Fine-tuned BERT model
- Custom dataset and data loader implementation
- GPU acceleration support
- Comprehensive evaluation metrics
- Visualization of results

#### How to Run:
```bash
python src/fakeNews_Classifer_BertModel.py
```

#### Output:
- Creates a `visuals_BERT` directory with timestamped subdirectories
- Saves training history, confusion matrices, and metrics
- Prints detailed evaluation metrics during execution

## Performance Comparison

### Naive Bayes
- Pros:
  - Faster training and inference
  - Lower computational requirements
  - Simpler implementation
- Cons:
  - Lower accuracy compared to BERT
  - Less effective with complex language patterns

### BERT
- Pros:
  - Higher accuracy (99.83% on test set)
  - Better handling of complex language patterns
  - More robust to variations in text structure
- Cons:
  - Longer training time
  - Higher computational requirements
  - Requires GPU for efficient training

## Output Directories

### visuals_NB/
Contains results from the Naive Bayes classifier:
- Confusion matrices
- Evaluation metrics
- Predictions
- Summary statistics

### visuals_BERT/
Contains results from the BERT classifier:
- Training history plots
- Confusion matrices
- Evaluation metrics
- Predictions
- Summary statistics

## Notes
- Both implementations use the same training data (`True.csv` and `Fake.csv`)
- Both use the same test data (`wellsFakeNewsClass.zip`)
- The BERT implementation requires more computational resources
- Results are saved in timestamped directories to prevent overwriting

## Troubleshooting
1. If you encounter memory issues with BERT:
   - Reduce the batch size in the code
   - Use a smaller model variant
   - Ensure you have enough GPU memory

2. If you encounter NLTK data issues:
   - The code will attempt to download required NLTK data
   - If it fails, manually download the required NLTK data

3. If you encounter CUDA/GPU issues:
   - The code will fall back to CPU if GPU is not available
   - Training will be slower but will still work
