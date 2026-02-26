# 🔍 Advanced Model Selection using GridSearchCV

This project demonstrates advanced model selection and hyperparameter tuning using GridSearchCV in Scikit-learn.

Multiple classification algorithms are compared systematically to determine the best-performing model and optimal hyperparameters using 5-fold cross-validation.

## 🚀 Objective

To:

Compare multiple ML classification models

Perform hyperparameter tuning using Grid Search

Evaluate models using cross-validation

Identify the best model based on mean CV accuracy

## 🧠 Models Compared

Logistic Regression

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Random Forest Classifier

## 🔎 Explanation

C (Logistic & SVC) → Regularization strength

kernel (SVC) → Type of decision boundary

n_neighbors (KNN) → Number of nearest neighbors

n_estimators (Random Forest) → Number of trees

## 🏗️ Model Selection Function

The Model_selection() function:

Iterates through models

Applies GridSearchCV

Performs 5-fold cross-validation

Extracts:

Best score

Best hyperparameters

Returns results as a Pandas DataFrame

## 📊 Workflow

Prepare dataset (x, y)

Define models list

Define hyperparameter grid

Call Model_selection()

Analyze best performing model

## 📈 Output

The function returns a DataFrame:

model_used	best_score	best_parameter
LogisticRegression	97.2	{'C': 20}
SVC	98.1	{'kernel': 'rbf', 'C': 30}
KNN	96.5	{'n_neighbors': 10}
RandomForest	99.0	{'n_estimators': 40}

(Example output — depends on dataset)

## 🛠️ Technologies Used

Python

Pandas

Scikit-learn

GridSearchCV

Cross-validation (cv=5)

## 🎯 Key Learnings

Grid Search exhaustively searches parameter combinations

Cross-validation prevents overfitting

Different models respond differently to hyperparameters

Model comparison should always be systematic

## 🏁 Conclusion

This implementation provides a clean and scalable way to:

✔ Compare multiple models
✔ Tune hyperparameters efficiently
✔ Select the most suitable model for a dataset

It demonstrates practical understanding of model evaluation, hyperparameter tuning, and structured ML experimentation.
