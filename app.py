import os
import streamlit as st
import numpy as np
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_visualizer import Visualizer

if 'proceed_1' not in st.session_state:
            st.session_state['proceed_1']=False

if 'proceed_2' not in st.session_state:
            st.session_state['proceed_2']=False

if 'proceed_3' not in st.session_state:
            st.session_state['proceed_3']=False

if 'proceed_4' not in st.session_state:
            st.session_state['proceed_4']=False

def handle_proceed_1():
	if st.session_state['proceed_1']==False:
		st.session_state['proceed_1']=True
def handle_proceed_2():
	if st.session_state['proceed_2']==False:
		st.session_state['proceed_2']=True
def handle_proceed_3():
	if st.session_state['proceed_3']==False:
		st.session_state['proceed_3']=True
def handle_proceed_4():
	if st.session_state['proceed_4']==False:
		st.session_state['proceed_4']=True



st.title("CSV to ML")
st.write('An auto EDA, Hyperparameter tuning & Prediction service for csv files')
uploaded_file=st.file_uploader("Choose a csv file")


if uploaded_file is not None:
	st.write("Presenting the basic EDA for the file")
	
	#Read the input csv file as pandas dataframe
	df=pd.read_csv(uploaded_file)

	#DataIngestion object
	obj=DataIngestion()
	train_data,test_data=obj.initiate_data_ingestion(df)  

	#Show the head of the data
	st.dataframe(df.head(5).style.highlight_max(axis=0))  

	#Data transformation object
	data_transformation=DataTransformation(train_data,test_data)

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
	#st.write(problem_type)

	proceed_1=st.button('Proceed',use_container_width=True, key=1, on_click=handle_proceed_1)
	if not proceed_1:
		st.warning('Proceed with training')

	if st.session_state["proceed_1"]:
		
		#Update Datatransformation object
		data_transformation.num_cat_columns=viz.num_cat_columns

		X_train,y_train,X_test,y_test=data_transformation.initiate_data_transformation(target)
		#st.write("Before initiate_model_trainer")
		#st.write(X_train.shape)
		modeltrainer=ModelTrainer(problem_type)
	
		if problem_type=='regression':
			best_model,models,best_score,scores=modeltrainer.initiate_model_trainer(X_train,y_train,X_test,y_test,modeltrainer.regression_param_space)
		else:
			best_model,models,best_score,scores=modeltrainer.initiate_model_trainer(X_train,y_train,X_test,y_test,modeltrainer.classification_param_space)

		st.write("Initial estimates suggests "+str(best_model)+" is best fitting the data")
		if problem_type=='regression':
			st.write("Report for the models used and their R2 Score")
		else:
			st.write("Report for the models used and their F1 Score")

		temp_df=pd.concat([pd.DataFrame(models),pd.DataFrame(scores)],axis=1, ignore_index=True)
		st.data_editor(temp_df.rename(columns = {0:'Model', 1:'Score'}))
		
		target_model = st.selectbox(
				'Select the model to proceed with hyperparameter tuning',
				models)


		proceed_2=st.button('Proceed',use_container_width=True,key=2, on_click=handle_proceed_2)
		if not proceed_2:
			st.warning('Proceed with tuning')
		
		if st.session_state["proceed_2"]:
			st.write("Here are the parameters to set for "+str(target_model))

			space=modeltrainer.get_param_space(target_model)
			slider_values=[]
			for param in space:
				values = st.slider(
					'Select a range of '+str(param),
					space[param][0], space[param][1], (2, 20),key=param)
				slider_values.append(values)
			
			proceed_3=st.button('Train',use_container_width=True,key=3,on_click=handle_proceed_3)
			if not proceed_3:
				st.warning('Proceed with Training')
				
			if st.session_state["proceed_3"]:
				best_score, best_param=modeltrainer.train(X_train,y_train,X_test,y_test,slider_values,target_model)
				
				st.write("Best HyperParameters are:")
				st.data_editor(best_param)
				if problem_type=='regression':
					st.write("Best R2 Score is "+str(best_score))
				else:
					st.write("Best Accuracy Score is "+str(best_score))

				proceed_4=st.button('Use the generated model',use_container_width=True,key=4, on_click=handle_proceed_4)
				if not proceed_4:
					st.warning('Proceed on using the model')

				if st.session_state["proceed_4"]:
					num_inputs,cat_inputs=viz.usr_input(df,numerical_columns,categorical_columns)
					output=modeltrainer.predict_output(df,num_inputs,cat_inputs,data_transformation.preprocessing_obj)
					if problem_type=='regression':
						st.write("Predicted "+str(target)+ " is "+str(output))
					else:
						result=df[target].unique()[np.array(output).argmax(axis=1)]
						st.write("Predicted "+str(target)+ " is "+str(result))


else:
    st.header("Please upload a csv file")

