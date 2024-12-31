import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


# -------------------------------------------------------------------------
# DataIngestionConfig: A dataclass to store the paths where the raw, train,
# and test datasets will be saved. The default values point to the "artifacts"
# directory, which is commonly used for storing intermediate or final artifacts
# of a data pipeline.
# -------------------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# -------------------------------------------------------------------------
# DataIngestion: This class handles reading the raw dataset, splitting it into
# train and test sets, and saving these sets (as well as the raw data) to the
# designated paths in the artifacts folder.
# -------------------------------------------------------------------------
class DataIngestion:
    def __init__(self):
        # Creates an instance of DataIngestionConfig, which defines where our
        # train, test, and raw data files will be written.
        self.ingestion_config = DataIngestionConfig()

    # -------------------------------------------------------------------------
    # initiate_data_ingestion:
    # 1. Logs the beginning of the data ingestion process.
    # 2. Reads the dataset from the specified CSV file.
    # 3. Saves the raw data to a location specified by raw_data_path.
    # 4. Splits the dataset into train/test sets.
    # 5. Saves the train/test sets to their respective paths in artifacts.
    # 6. Returns the paths to the train/test CSV files.
    # -------------------------------------------------------------------------
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reads the CSV file into a pandas DataFrame. 
            # Update the path to your actual data as needed.
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # Ensure the directory for train/test artifacts exists. If not, create it.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw DataFrame to the raw_data_path for reference or backup.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            # Perform an 80/20 train/test split with a fixed random_state for reproducibility.
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Write the train set to its designated path.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Write the test set to its designated path.
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the paths for the train and test CSV files.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # If an exception occurs, wrap it in a CustomException (likely for logging
            # or further handling) and re-raise it.
            raise CustomException(e, sys)


# -------------------------------------------------------------------------
# When run as a standalone script, this block will instantiate the DataIngestion
# class and initiate the data ingestion process, providing train and test file paths.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))