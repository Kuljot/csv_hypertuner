import os
import sys

import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        pass
    
    def initiate_data_ingestion(self,df:pd.DataFrame)-> {pd.DataFrame,pd.DataFrame}:
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=0)
            logging.info("Ingestion of the data is completed")
            return train_set,test_set
        except Exception as e:
            raise CustomException(e,sys)
    