import pandas as pd
import streamlit as st

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
            