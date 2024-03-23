# Customer Fraud Detection

This repository contains a comprehensive Python-based analysis and machine learning workflow for detecting fraudulent activities among customers based on their transaction details and customer profiles. The project involves data loading, preprocessing, exploratory data analysis, feature engineering, and the application of various machine learning models to identify potential fraud.

## Overview

The project is structured into a single, detailed script that covers the following key steps:

1. **Data Loading**: Importing customer profile and transaction datasets.
2. **Data Preprocessing**: Cleaning data, handling missing values, and renaming columns for better readability.
3. **Exploratory Data Analysis (EDA)**: Analyzing the datasets to understand the underlying patterns, including counting unique values and plotting various distributions.
4. **Feature Engineering**: Creating new features that can help in the detection of fraudulent activities, such as the total transaction amount, average transaction amount, and counts of failed transactions.
5. **Model Training and Evaluation**: Training machine learning models like Support Vector Machines (SVM), Decision Trees, Random Forests, and XGBoost, and evaluating their performance in detecting fraud.
6. **Model Tuning**: Fine-tuning models using GridSearchCV and RandomizedSearchCV to improve performance.
7. **Visualization**: Visualizing data distributions, feature correlations, and model performance metrics to gain insights.

## Prerequisites

Ensure you have Python 3.8+ installed on your system. The project depends on several Python libraries including:

- pandas
- numpy
- sklearn
- matplotlib
- seaborn

You can install these dependencies via pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
