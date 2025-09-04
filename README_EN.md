# KNN Algorithm Implementation

A K-Nearest Neighbors (KNN) classification algorithm implementation based on scikit-learn, demonstrated using the Iris dataset.

## Project Overview

This project implements a complete KNN classifier including data preprocessing, model training, hyperparameter optimization, and performance evaluation.

## Features

- Classification using the Iris dataset
- Data standardization preprocessing
- Train/test set splitting
- Grid search for hyperparameter optimization
- 3-fold cross-validation
- Model performance evaluation

## Tech Stack

- Python 3.x
- pandas - Data manipulation
- scikit-learn - Machine learning library
- StandardScaler - Data standardization
- GridSearchCV - Hyperparameter optimization

## Algorithm Workflow

1. **Data Loading**: Uses sklearn's built-in Iris dataset
2. **Data Preprocessing**: Standardizes features using StandardScaler
3. **Data Splitting**: 80% training set, 20% test set
4. **Model Training**: Uses KNeighborsClassifier
5. **Hyperparameter Optimization**: Grid search for optimal k values (5, 10, 15)
6. **Cross Validation**: 3-fold cross-validation for model stability
7. **Performance Evaluation**: Calculates accuracy and outputs prediction results

## Usage

```bash
python KNN.py
```

## Output

The program outputs the following information:
- Original data and standardized data
- Comparison between predictions and true labels
- Best parameter configuration
- Model accuracy score

## File Description

- `KNN.py` - Main algorithm implementation
- `KNN algorithm logic.xlsx` - Algorithm logic documentation
- `README.md` - Main project documentation (English)
- `README_CN.md` - Chinese project documentation
- `README_EN.md` - English project documentation