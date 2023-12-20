import os
import sys
import streamlit as st
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import(
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.metrics import r2_score,accuracy_score
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

class my_model(BaseEstimator):
    def __init__(self, model_type:str=''):
        """
        A Custome BaseEstimator that can switch between classifiers.
        :param classifier_type: string - The switch for different classifiers
        """
        self.model_type = model_type

    def select_model(self):
        #Classification Models
        if self.model_type == 'DecisionTreeClassifier':
            self.model_ = DecisionTreeClassifier()
        elif self.model_type == 'KNeighborsClassifier':
            self.model_ = KNeighborsClassifier()
        elif self.model_type == 'XGBClassifier':
            self.model_ = XGBClassifier()
        elif self.model_type == 'RandomForestClassifier':
            self.model_ = RandomForestClassifier()
        #Regression Models
        elif self.model_type == 'RandomForestRegressor':
            self.model_ = RandomForestRegressor()
        elif self.model_type == 'DecisionTreeRegressor':
            self.model_ = DecisionTreeRegressor()
        elif self.model_type == 'KNeighborsRegressor':
            self.model_ = KNeighborsRegressor()
        elif self.model_type == 'XGBRegressor':
            self.model_ = XGBRegressor()
        else:
            raise ValueError('Unkown classifier type.')

        return self.model_

    def fit(self, X, y=None):
        self.select_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.model_.predict(X)
    def predict_proba(self, X):
        return self.model_.predict_proba(X)


    def score(self, X, y):
        return self.model_.score(X, y)

class ModelTrainer:
    def __init__(self,problem_type:str):
        self.problem_type:str=problem_type
        self.pipeline=Pipeline([
            ('cls',my_model())
        ])
        self.classification_param_space={
            'cls__model_type':['DecisionTreeClassifier','KNeighborsClassifier','XGBClassifier','RandomForestClassifier']
        }
        
        self.regression_param_space={
            'cls__model_type':['DecisionTreeRegressor','XGBRegressor','RandomForestRegressor','KNeighborsRegressor']
        }

        self.model_params={
            'DecisionTreeClassifier':{'max_depth':[0,10],'min_samples_leaf':[0,10]},
            'KNeighborsClassifier':{'leaf_size':[0,10],'n_neighbors':[0,10]},
            'XGBClassifier':{'n_estimators':[0,10],'max_depth':[0,10],'min_samples_leaf':[0,10]},
            'RandomForestClassifier':{'n_estimators':[0,10],'max_depth':[0,10],'min_samples_leaf':[0,10]},
            'RandomForestRegressor':{'max_depth':[2,10],'min_samples_leaf':[0,10]},
            'DecisionTreeRegressor':{'max_depth':[0,10],'max_features':[0,10],'max_leaf_nodes':[0,10]},
            'KNeighborsRegressor':{'n_neighbors':[0,10],'leaf_size':[0,10]},
            'XGBRegressor':{'n_estimators':[0,10],'max_depth':[0,10],'min_samples_leaf':[0,10]}
        }

    def initiate_model_trainer(self,x_train,y_train,x_test,y_test,param_space):
        try:
            if self.problem_type=='regression':
                search = GridSearchCV(self.pipeline , param_space, n_jobs=-1, cv=2,scoring='r2')
            else:
                search = GridSearchCV(self.pipeline , param_space, n_jobs=-1, cv=2,scoring='accuracy')
            search_result=search.fit(x_train, y_train)

            best_model_score=search_result.best_score_
            best_model=search_result.best_params_

            if best_model_score<0.0:
                raise CustomException("No best model found")
            logging.info(f"Best Model found on both training and testing dataset")

            return best_model["cls__model_type"],param_space["cls__model_type"],best_model_score,search.cv_results_["mean_test_score"]
        except Exception as e:
            raise CustomException(e,sys)

    def get_param_space(self,model:str):
        return self.model_params[model]


    def train(self,x_train,y_train,x_test,y_test,slider_values,model:str):

        try:
            base_estimator=my_model(model_type=model).select_model()
            
            param_grid={}
            for i,param in enumerate(self.model_params[model]):
                param_grid[param]=[]
                param_grid.update(
                    {
                        param : [slider_values[i][0],(slider_values[i][0]+slider_values[i][1])//2,slider_values[i][1]]
                    }
                )
            
            if self.problem_type=='regression':
                search = GridSearchCV(estimator=base_estimator , param_grid=param_grid, n_jobs=-1, cv=2,scoring='r2')
            else:
                search = GridSearchCV(estimator=base_estimator , param_grid=param_grid, n_jobs=-1, cv=2,scoring='accuracy')
    
            search_result=search.fit(x_train, y_train)

            best_model_score=search_result.best_score_
            best_params=search_result.cv_results_["params"][search_result.best_index_]
            self.best_model=search_result.best_estimator_

            if best_model_score<0.0:
                raise CustomException("No best model found")
            logging.info(f"Best Model found on both training dataset")
            
            #Fit the final model
            self.best_model.fit(x_train,y_train)
            y_pred=self.best_model.predict(x_test)
            if self.problem_type=='regression':
                best__score=r2_score(y_test,y_pred)
            else:
                best__score=accuracy_score(y_test,y_pred)
            
            return best__score, best_params
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict_output(self,df,num_inputs:dict,cat_inputs:dict,preprocessor):
        model=self.best_model
        X={}
        for column in df.columns:
            if column in num_inputs:
                X[column]=[]
                X.update({column:num_inputs[column]})
            elif column in cat_inputs:
                X[column]=[]
                X.update({column:cat_inputs[column]})
        _df=pd.DataFrame(X,index=[0])

        #Pass through transformer
        processed_data=preprocessor.transform(_df)


        return model.predict(processed_data)
