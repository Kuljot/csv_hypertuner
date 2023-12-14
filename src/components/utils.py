import sys
import os
import dill
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        #st.data_editor(y_train)
        for i in range(len(list(models))):
            model=list(models.values())[i]
            # model=DecisionTreeClassifier()
            st.write(list(models.values())[i])
            try:
                st.write("Training model...")
                st.write(model.fit(x_train,y_train))
                st.write("Model training completed.")

            except Exception as e:
                raise CustomException(e,sys)

            
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]= test_model_score
            print(report)
            st.write(report)
        return report
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

class Visualizer:
    def __init__(self,num_columns,cat_columns):
        self.num_columns=num_columns
        self.cat_columns=cat_columns
        self.num_cat_columns=[]

    def outliers_count(self,df:pd.DataFrame):
        quartiles = df.quantile([0.25, 0.75])
        IQR = quartiles[0.75] - quartiles[0.25]
        higher_outlier = df.quantile(0.75) + (IQR * 1.5)
        lower_outlier = df.quantile(0.25) - (IQR * 1.5)

        num_outliers = df[(df < lower_outlier) | (df > higher_outlier)].shape[0]
        return num_outliers

        
    def num_visualize(self,df:pd.DataFrame,columns:list):
        self.frame=pd.DataFrame({
                "columns":columns,
                "nullval":[df[column].isnull().sum()/df.shape[0]*100 for column in columns],
                "outliers":[self.outliers_count(df[column])/df.shape[0]*100 for column in columns],
                #"categorical":[False for column in columns]
            })
        st.data_editor(
            self.frame,
            column_config={ 
                "nullval": st.column_config.ProgressColumn(
                    "Missing Values",
                    help="Number of missing values",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                ),
                "outliers": st.column_config.ProgressColumn(
                    "Outliers in data",
                    help="Number of outlier values",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                ),
            },disabled=["columns"],hide_index=True,
        )
        



    def cat_visualize(self,df:pd.DataFrame,columns:list):
        st.data_editor(
            pd.DataFrame({
                "columns":columns,
                "unique":[df[column].unique().size for column in columns],
                "nullval":[df[column].isnull().sum()/df.shape[0]*100 for column in columns]
            }),
            column_config={
                "nullval": st.column_config.ProgressColumn(
                    "Missing Values",
                    help="Number of missing values",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                ),
            },disabled=["columns"],hide_index=True,
        )

    def target_col(self,df:pd.DataFrame):
        problem_type=''
        #Select the categorical columns
        self.num_cat_columns = st.multiselect(
            'Select the columns from numerical columns that are categorical in nature',
            self.num_columns)
        #Select the target column
        st.write("Choose the Target Column")
        target = st.selectbox(
            'Select the target column',
            (df.columns))

        st.write("You selected:", target)
        if target in self.num_columns:
            if target not in self.num_cat_columns:
                st.write("It is a regression problem")
                problem_type='regression'
            else:
                st.write("It is  a classification problem")
                problem_type='classification'
        else:
            st.write("It is a classification problem")
            problem_type='classification'

        return target,problem_type

