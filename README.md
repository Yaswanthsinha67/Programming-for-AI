# Programming-for-AI

## Fake News Detection Project

A machine learning and NLP-based project to classify news articles as real or fake using both rule-based and machine learning approaches.

### Project Overview

This project implements a comprehensive fake news detection system using the WELFake Dataset. It compares two classification approaches:
1. **Rule-Based Classifier**: Uses predefined keywords commonly found in fake news
2. **Machine Learning Classifier**: Uses Naive Bayes with TF-IDF feature extraction

### Features

- **Data Processing**: Comprehensive text preprocessing including:
  - URL and HTML tag removal
  - Punctuation and number removal
  - Stopword removal
  - Lemmatization

- **Feature Extraction**: TF-IDF vectorization with n-grams (unigrams and bigrams)

- **Classification Models**:
  - Rule-based approach using keyword matching
  - Multinomial Naive Bayes classifier

- **Evaluation Metrics**:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix visualization
  - ROC-AUC Score

### Files

- `12104.ipynb` - Main Jupyter notebook containing the complete analysis and model implementation
- `WELFake_Dataset.csv` - Dataset containing news articles with labels (0: Real, 1: Fake)
- `README.md` - Project documentation

### Requirements

The project uses the following libraries:
- **Data Processing**: pandas, numpy
- **NLP**: nltk (with stopwords and wordnet)
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn

### Workflow

1. **Data Loading**: Load the WELFake dataset from CSV
2. **Data Inspection**: Analyze dataset shape, columns, and label distribution
3. **Preprocessing**: Handle missing values and combine title + text fields
4. **Text Cleaning**: Apply preprocessing function to clean and normalize text
5. **Train-Test Split**: Split data into 80% training and 20% testing sets
6. **Feature Extraction**: Convert text to TF-IDF vectors
7. **Model Training**: Train both rule-based and ML classifiers
8. **Evaluation**: Compare performance using multiple metrics
9. **Visualization**: Display confusion matrix and identify top fake news indicators

### Key Findings

- Compares the effectiveness of rule-based vs. machine learning approaches
- Identifies top TF-IDF features that indicate fake news
- Provides comprehensive evaluation metrics for model comparison

### Usage

Run the Jupyter notebook `12104.ipynb` in Google Colab (requires Google Drive access for dataset) or locally with the dataset file in the correct path.

### Author

Yaswanthsinha67