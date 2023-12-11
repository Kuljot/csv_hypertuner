import os
import sys
from logger import logging
from exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass #Special decorator for storing data and deried data e.g. area
class DataIngestionConfig:
    train_data_path:str =os.path.join('artifacts',"train.csv")
    test_data_path:str =os.path.join('artifacts',"test.csv")
    raw_data_path:str =os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self,df):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Recieved dataset as dataframe saving as csv")

            #Make folder for keeping data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            #save the csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            #Split data
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=4)

            #Save as test & train
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
    