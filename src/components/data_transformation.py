import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# --------------------------------------------------------------------------------
# DataTransformationConfig: Stores the file path where the preprocessing object
# (which includes transformations/pipelines) will be saved as a .pkl file.
# --------------------------------------------------------------------------------
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

# --------------------------------------------------------------------------------
# DataTransformation:
#   - Responsible for creating preprocessing pipelines for both numerical and
#     categorical features.
#   - Applies these pipelines to training and test datasets.
#   - Saves the fitted pipeline object for later use in model inference or future
#     data transformations.
# --------------------------------------------------------------------------------
class DataTransformation:
    def __init__(self):
        # Initializes a config object that defines where the processor .pkl file will be saved.
        self.data_transformation_config = DataTransformationConfig()

    # --------------------------------------------------------------------------------
    # get_data_transformer_object:
    #   - Creates and returns a ColumnTransformer that preprocesses numerical
    #     and categorical columns differently.
    #   - For numeric columns: Imputes missing values with median, then scales them.
    #   - For categorical columns: Imputes missing values with most frequent class,
    #     then applies one-hot encoding and scales them.
    # --------------------------------------------------------------------------------
    def get_data_transformer_object(self):
        """
        This function is responsible for creating and returning
        a data transformer (ColumnTransformer) that will
        handle numerical and categorical features appropriately.
        """
        try:
            # Specify which columns are numerical and which are categorical.
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline: median imputation + standard scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline: most frequent imputation + one-hot encoding + standard scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # ColumnTransformer: apply 'num_pipeline' to numerical_columns and
            # 'cat_pipeline' to categorical_columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            # Wrap and re-raise exceptions in CustomException for better tracing.
            raise CustomException(e, sys)
        
    # --------------------------------------------------------------------------------
    # initiate_data_transformation:
    #   - Reads train and test CSVs into DataFrames.
    #   - Creates a preprocessing object (ColumnTransformer) via get_data_transformer_object.
    #   - Applies the transformer to both train and test feature sets.
    #   - Combines transformed features with the target column for both train and test.
    #   - Saves the preprocessing object as a .pkl file for future use.
    #   - Returns the transformed train/test arrays and the file path to the saved object.
    # --------------------------------------------------------------------------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the train and test data from CSV files.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Create the ColumnTransformer using get_data_transformer_object.
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate the target column from the features in train and test sets.
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit the ColumnTransformer on the training features, then transform them.
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # Transform the test features (without fitting again).
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target variable.
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Save the fitted preprocessing object (pipeline) so it can be used for inference.
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed arrays and the preprocessor file path.
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
