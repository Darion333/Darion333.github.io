---
layout: post
comments: true
title: Tutorial 2. Training the Machine Learning Model
---

Definition of Classification vs Regression problem. 
  Give examples about Classification vs Regression tasks.
  This is a classification task.
Commonly used classification algorithms (explain each idea and point the readers to online resource to learn more about the methods):
Logistic regression. 
  Random forest.
  Support Vector Machine
How to train an ML model?
  Training set vs test set
    What we need them
    How to split them (reasonable ratio)
  Evaluation the perform of classification task: 
    accuracy, TP, FP, TN, FN, ROC curve etc.
    Performance on the training set vs performance on the test set
Show the code to train the ML models
  Introduce Scikit learn
  Explain which function does the train/test splitting
  Explain which function performs the training

## Definition of Classification vs. Regression Problems
Classification and regression are two fundamental types of supervised machine learning tasks. Classification involves predicting a discrete label or category. For example, identifying if an email is "spam" or "not spam" or determining whether a tumor is "benign" or "malignant." Regression, on the other hand, predicts continuous values, such as forecasting house prices or estimating a person's weight based on their height.

## Examples of Classification vs. Regression Tasks
Classification Example: Determining if a molecule is cancerous or non-cancerous.
Regression Example: Predicting the boiling point of a chemical compound based on its structure.
This is a Classification Task
In our case, distinguishing between cancerous and non-cancerous PAH molecules is a classification task. The model learns to assign a molecule to one of these two categories based on its features.

## Commonly Used Classification Algorithms
Logistic Regression: This algorithm models the probability that an input belongs to a particular category. It uses a logistic function to squeeze predicted values between 0 and 1, making it effective for binary classification problems. To learn more, visit Logistic Regression - Scikit-Learn.

Random Forest: This is an ensemble method that builds multiple decision trees during training and combines their outputs for improved accuracy and robustness. It reduces overfitting and handles large datasets efficiently. More details can be found here: Random Forest - Scikit-Learn.


Support Vector Machine (SVM): SVMs classify data by finding the optimal hyperplane that separates different classes with the largest margin. It’s particularly effective for high-dimensional spaces. Learn more at Support Vector Machines - Scikit-Learn.


These algorithms are widely used in classification tasks and can be explored further through these resources.

















