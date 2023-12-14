import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.components.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    preprocessor_target_file_path=os.path.join('artifacts','preprocessor_target.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.numerical_columns=[]
        self.categorical_columns=[]
        self.num_cat_columns=[]
        self.target=[]
        self.test_path=os.path.join('artifacts','test.csv')
        self.train_path=os.path.join('artifacts','train.csv')
    
    def get_column_types(self,df:pd.DataFrame):

        for column in df.columns:
            if df[column].dtype =='object':
                self.categorical_columns.append(column)
            else:
                self.numerical_columns.append(column)
        return self.numerical_columns, self.categorical_columns
    
    def get_data_transformer_object(self,target_col:bool):
        '''
        Function for data transformation
        '''
        try:
            problem_type=''
            #Remove the num_cat columns from num_columns and add it to the categorical columns
            for column in self.numerical_columns:
                if column in self.num_cat_columns:
                    self.numerical_columns.remove(column)
                    self.categorical_columns.append(column)
                    print(column)

            #Remove the target column from lists
            if self.target in self.numerical_columns:
                self.numerical_columns.remove(self.target)
                problem_type='regression'
            if self.target in self.categorical_columns:
                self.categorical_columns.remove(self.target)
                problem_type='classification'

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
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore',drop='first')),
                    # ("scaler",StandardScaler(with_mean=False))
                ]
            )

            target_pipeline=[]
            if problem_type =='regression':
               target_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    ]
                )
            else:
                target_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ]
                )

            

            logging.info("Categorical columns encoding completed")

            

            # preprocessor=ColumnTransformer(
            #     [
            #         ("num_pipeline",num_pipeline,self.numerical_columns),
            #         ("cat_pipeline",cat_pipeline,self.categorical_columns),
            #         ("target_pipeline",target_pipeline,[self.target])
            #     ]
            # )
            if target_col:
                preprocessor=ColumnTransformer(
                    [
                        ("target_pipeline",target_pipeline,[self.target])
                    ]
                )
            else:
                preprocessor=ColumnTransformer(
                    [
                        ("num_pipeline",num_pipeline,self.numerical_columns),
                        ("cat_pipeline",cat_pipeline,self.categorical_columns),
                    ]
                )


            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,target):

        try:
            self.target=target
            train_df=pd.read_csv(self.train_path)
            test_df =pd.read_csv(self.test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object(False)
            preprocessing_obj_target=self.get_data_transformer_object(True)

            x_train=train_df.drop(columns=[self.target],axis=1)
            y_train=pd.DataFrame(np.expand_dims(train_df[self.target].values, axis=1),columns=[self.target])
            

            x_test=test_df.drop(columns=[self.target],axis=1)
            y_test = pd.DataFrame(np.expand_dims(test_df[self.target].values, axis=1),columns=[self.target])


            logging.info(
                f"Applying preprocessing object on training and testing df"
            )
            print("Train DataFrame Columns:", train_df.columns)
            print("Test DataFrame Columns:", test_df.columns)

            #After Transformation
            x_train=preprocessing_obj.fit_transform(x_train)
            x_test=preprocessing_obj.fit_transform(x_test)

            #st.data_editor(y_train)
            y_train=preprocessing_obj_target.fit_transform(y_train)
            #st.data_editor(y_train)
            y_test=preprocessing_obj_target.fit_transform(y_test)
            
            logging.info(f"Saved preprocessing object.")
            st.write(sys.version)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_target_file_path,
                obj=preprocessing_obj_target
            )

            return(
                x_train,
                y_train,
                x_test,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.preprocessor_target_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
