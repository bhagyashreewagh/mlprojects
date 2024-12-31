import os
import sys
import pickle
import logging  # Ensure logging is imported
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f"Error while saving object to {file_path}: {e}", sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV and returns their test R2 scores.
    """
    try:
        if not models:
            raise CustomException("No models provided for evaluation.")

        report = {}

        for model_name, model in models.items():
            try:
                logging.info(f"Evaluating model: {model_name}")
                params = param.get(model_name, {})
                if not params:
                    logging.info(f"No hyperparameters specified for {model_name}. Using default settings.")

                # Perform grid search
                logging.info(f"Grid search for {model_name} with params: {params}")
                gs = GridSearchCV(model, params, cv=3)
                gs.fit(X_train, y_train)

                # Update model with best parameters
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # R2 Scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                # Log model performance
                logging.info(f"{model_name} - Train R2: {train_model_score}, Test R2: {test_model_score}")

                report[model_name] = test_model_score

            except Exception as e:
                raise CustomException(
                    f"Error while evaluating model {model_name}: {e}", sys
                )

        return report

    except Exception as e:
        raise CustomException(f"Error in evaluating models: {e}", sys)

def load_object(file_path):
    """
    Loads a Python object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(f"Error while loading object from {file_path}: {e}", sys)
