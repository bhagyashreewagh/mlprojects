import os
import sys

import numpy as np
import pandas as pd
import dill  # Although dill is imported, it's not being used in this snippet
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# --------------------------------------------------------------------------------
# save_object:
#     - Creates the directory (if it doesn’t exist) specified by file_path
#       and saves a given Python object as a .pkl file using pickle.
# --------------------------------------------------------------------------------
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Wrap and re-raise any exceptions as a CustomException for consistent handling
        raise CustomException(e, sys)
    

# --------------------------------------------------------------------------------
# evaluate_models:
#     - Takes training and testing data, a dictionary of models, and a dictionary
#       of hyperparameter grids (param).
#     - For each model:
#         1) Performs a GridSearchCV to find the best hyperparameters.
#         2) Re-trains the model using the best params.
#         3) Predicts on both train and test sets.
#         4) Computes the R2 score for train and test sets.
#         5) Stores the test score in a 'report' dictionary keyed by model name.
#     - Returns a dictionary where keys are model names and values are test R2 scores.
# --------------------------------------------------------------------------------
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        # Loop through each model in the provided dictionary
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            # Perform grid search with the given hyperparameters (param) for this model
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Update the model with best hyperparameters and retrain
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on both training and testing sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores for train and test
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

# --------------------------------------------------------------------------------
# load_object:
#     - Loads a previously saved Python object from a .pkl file using pickle.
# --------------------------------------------------------------------------------
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
