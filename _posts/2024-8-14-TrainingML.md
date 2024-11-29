---
layout: post
comments: true
title: Tutorial 3. Training the Machine Learning Model
---


## Definition of Classification vs. Regression Problems
Classification and regression are two fundamental types of supervised machine learning tasks. Classification involves predicting a discrete label or category. For example, identifying if an email is "spam" or "not spam" or determining whether a tumor is "benign" or "malignant." Regression, on the other hand, predicts continuous values, such as forecasting house prices or estimating a person's weight based on their height.

## Examples of Classification vs. Regression Tasks
Classification Example: Determining if a molecule is cancerous or non-cancerous.
Regression Example: Predicting the boiling point of a chemical compound based on its structure.
This is a Classification Task
In our case, distinguishing between cancerous and non-cancerous PAH molecules is a classification task. The model learns to assign a molecule to one of these two categories based on its features.

## Commonly Used Classification Algorithms
Logistic Regression: This algorithm models the probability that an input belongs to a particular category. It uses a logistic function to squeeze predicted values between 0 and 1, making it effective for binary classification problems. To learn more, visit [Logistic Regression - Scikit-Learn](:https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)

Random Forest: This is an ensemble method that builds multiple decision trees during training and combines their outputs for improved accuracy and robustness. It reduces overfitting and handles large datasets efficiently. More details can be found here: [Random Forest - Scikit-Learn](:https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


Support Vector Machine (SVM): SVMs classify data by finding the optimal hyperplane that separates different classes with the largest margin. It’s particularly effective for high-dimensional spaces. Learn more at [Support Vector Machines - Scikit-Learn](:https://scikit-learn.org/1.5/modules/svm.html)


These algorithms are widely used in classification tasks and can be explored further through these resources.

## How to Train a Machine Learning Model
Training a machine learning (ML) model requires dividing your data into two essential subsets: the training set and the test set. The training set is used to teach the model, while the test set evaluates its generalization to unseen data. A common and effective split ratio is 80% for training and 20% for testing. For classification tasks, metrics such as accuracy, true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) offer insights into the model’s performance, with the ROC curve providing a deeper understanding of trade-offs between sensitivity and specificity. Comparing training set performance to test set performance reveals if the model is overfitting or underfitting.















