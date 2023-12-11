import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd 


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.numerical_columns=[]
        self.categorical_columns=[]
        self.num_cat_columns=[]
    
    def get_column_types(self,df:pd.DataFrame):

        for column in df.columns:
            if df[column].dtype =='object':
                self.categorical_columns.append(column)
            else:
                self.numerical_columns.append(column)
        return self.numerical_columns, self.categorical_columns
    
    def get_data_transformer_object(self):
        '''
        Function for data transformation
        '''
        try:

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical columns scaling completed")


            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding completed")

            #Remove the num_cat columns from num_columns and add it to the categorical columns
            for column in self.numerical_columns:
                if column in self.num_cat_columns:
                    self.numerical_columns.remove(column)
                    self.categorical_columns.append(column)
                    print(column)

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,self.numerical_columns),
                    ("cat_pipeline",cat_pipeline,self.categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"

            numerical_columns=["writing_score","reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training and testing df"
            )
            print("Train DataFrame Columns:", train_df.columns)
            print("Test DataFrame Columns:", test_df.columns)
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
