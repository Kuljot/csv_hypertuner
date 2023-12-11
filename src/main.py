import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from utils import Visualizer

st.title("CSV to ML")
st.write('An auto EDA, Hyperparameter tuning & Prediction service for csv files')
uploaded_file=st.file_uploader("Choose a csv file")

if uploaded_file is not None:
	st.write("Presenting the basic EDA for the file")
	#read the input csv file as pandas dataframe
	df=pd.read_csv(uploaded_file)

	#DataIngestion object
	obj=DataIngestion()
	train_data,test_data=obj.initiate_data_ingestion(df)  

	#Show the head of the data
	st.dataframe(df.head(5).style.highlight_max(axis=0))  

	#Data transformation object
	data_transformation=DataTransformation()
	#Check the dtypes of the columns
	numerical_columns,categorical_columns=data_transformation.get_column_types(df)

	#Visualizer object
	viz=Visualizer(numerical_columns,categorical_columns)
	#Show the numerical and categorical columns
	st.write("Numerical Columns")
	viz.num_visualize(df,numerical_columns)

	#Show Categorical columns
	st.write("Categorical Columns")
	viz.cat_visualize(df,categorical_columns)

	#Select the Target column
	target,problem_type=viz.target_col(df)

	proceed=st.button('Proceed')
	if not proceed:
		st.warning('Proceed with training')
		st.stop()
	st.success('Doing data transformation')
	
	data_transformation.num_cat_columns=viz.num_cat_columns
	transformer =data_transformation.get_data_transformer_object()

	#modeltrainer=ModelTrainer()
	# print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

	
else:
    st.header("Please upload a csv file")

    
