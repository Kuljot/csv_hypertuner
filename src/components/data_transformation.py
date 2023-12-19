import os
import sys
import numpy as np 
import pandas as pd 
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self,train_data:pd.DataFrame,test_data:pd.DataFrame):
        self.numerical_columns:list=[]
        self.categorical_columns:list=[]
        self.num_cat_columns:list=[]
        self.target:str=[]
        self.train_data:pd.DataFrame=train_data
        self.test_data:pd.DataFrame=test_data
        self.preprocessing_obj=None
        self.preprocessing_obj_target=None
    
    def get_column_types(self,df:pd.DataFrame)->{list, list}:
        for column in df.columns:
            if df[column].dtype =='object':
                self.categorical_columns.append(column)
            else:
                self.numerical_columns.append(column)
        return self.numerical_columns, self.categorical_columns
    
    

    def get_data_transformer(self,target_col:bool):
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
            
            logging.info("Categorical columns encoding completed")
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",MinMaxScaler())
                ]
            )
            logging.info("Numerical columns scaling completed")


            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore',drop='first'))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,self.numerical_columns),
                    ("cat_pipeline",cat_pipeline,self.categorical_columns),
                ]
            )


            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def get_target_transformer(self,target_col:bool):
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

            if self.target in self.numerical_columns:
                self.numerical_columns.remove(self.target)
                problem_type='regression'
            if self.target in self.categorical_columns:
                self.categorical_columns.remove(self.target)
                problem_type='classification'

            target_pipeline=[]
            
            if problem_type =='regression':
               target_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ]
                )
            else:
                target_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ]
                )
            
            
            preprocessor=ColumnTransformer(
                [
                    ("target_pipeline",target_pipeline,[self.target])
                ]
            )
            


            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    

    def initiate_data_transformation(self,target:str):

        try:
            self.target=target
            train_df=self.train_data
            test_df =self.test_data

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            self.preprocessing_obj=self.get_data_transformer(False)
            self.preprocessing_obj_target=self.get_target_transformer(True)

            X_train=train_df.drop(columns=[self.target],axis=1)
            Y_train=pd.DataFrame(np.expand_dims(train_df[self.target].values, axis=1),columns=[self.target])
            

            X_test=test_df.drop(columns=[self.target],axis=1)
            Y_test = pd.DataFrame(np.expand_dims(test_df[self.target].values, axis=1),columns=[self.target])


            logging.info(
                f"Applying preprocessing object on training and testing df"
            )
        
            x_train=self.preprocessing_obj.fit_transform(X_train)
            x_test=self.preprocessing_obj.fit_transform(X_test)
            
            
            y_train=self.preprocessing_obj_target.fit_transform(Y_train)
            y_test=self.preprocessing_obj_target.fit_transform(Y_test)
            logging.info(f"Saved preprocessing object.")
            

            return(
                x_train,
                y_train,
                x_test,
                y_test,
            )
        except Exception as e:
            raise CustomException(e,sys)