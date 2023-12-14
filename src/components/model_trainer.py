import os
import sys
import streamlit as st
from dataclasses import dataclass

from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.ensemble import(
    AdaBoostRegressor,
    AdaBoostClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression,SGDClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

from components.utils import save_object
from components.utils import evaluate_model

class my_model(BaseEstimator):
    def __init__(self, model_type:str='DecisionTreeClassifier'):
        """
        A Custome BaseEstimator that can switch between classifiers.
        :param classifier_type: string - The switch for different classifiers
        """
        self.model_type = model_type


    def fit(self, X, y=None):
        if self.model_type == 'DecisionTreeClassifier':
            self.model_ = DecisionTreeClassifier()
        elif self.model_type == 'KNeighborsClassifier':
            self.model_ = KNeighborsClassifier()
        elif self.model_type == 'XGBClassifier':
            self.model_ = XGBClassifier()
        elif self.model_type == 'GradientBoostingClassifier':
            self.model_ = GradientBoostingClassifier()
        #Regression Models
        elif self.model_type == 'RandomForestRegressor':
            self.model_ = RandomForestRegressor()
        elif self.model_type == 'DecisionTreeRegressor':
            self.model_ = DecisionTreeRegressor()
        elif self.model_type == 'LinearRegression':
            self.model_ = LinearRegression()
        elif self.model_type == 'KNeighborsRegressor':
            self.model_ = KNeighborsRegressor()
        elif self.model_type == 'XGBRegressor':
            self.model_ = XGBRegressor()
        else:
            raise ValueError('Unkown classifier type.')

        self.model_.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.model_.predict(X)
    def predict_proba(self, X):
        return self.model_.predict_proba(X)


    def score(self, X, y):
        return self.model_.score(X, y)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.pipeline=Pipeline([
            ('cls',my_model())
        ])
        self.classification_param_space={
            'cls__model_type':['DecisionTreeClassifier','KNeighborsClassifier','XGBClassifier','GradientBoostingClassifier']
        }
        
        self.regression_param_space={
            'cls__model_type':['RandomForestRegressor','DecisionTreeRegressor','LinearRegression','KNeighborsRegressor','XGBRegressor']
        }

        # self.reg_models={
        #         "Random Forest": RandomForestRegressor(),
        #         "Decision Tree": DecisionTreeRegressor(),
        #         #"Gradient Boosting": GradientBoostingRegressor(),
        #         "Linear Regression": LinearRegression(),
        #         "K-Neighbors Regression":KNeighborsRegressor(n_neighbors=3),
        #         "XGBRegression": XGBRegressor(),
        #         #"CatBoosting Regression": CatBoostRegressor(verbose=False),
        #         # "AdaBoosting Regression": AdaBoostRegressor()
        #     }
        # self.cls_models={
        #         #"Random Forest": RandomForestClassifier(max_depth=2, random_state=0),
        #         # "Decision Tree": DecisionTreeClassifier(),
        #         # #"Gradient Boosting": GradientBoostingClassifier(),
        #         # #"SGDC Classifier": SGDClassifier(),
        #         # "K-Neighbors Classifier":KNeighborsClassifier(n_neighbors=3),
        #         # "XGBClassifier": XGBClassifier(),
        #         # #"CatBoosting Classifier": CatBoostClassifier(verbose=False),
        #         # "AdaBoosting Classifier": AdaBoostClassifier()
        #     }
    
    def initiate_model_trainer(self,x_train,y_train,x_test,y_test,param_space):
        try:
            st.write(param_space)
            logging.info("Split training and test input data")
            st.write(x_train.shape)
            st.write(y_train.shape)
            st.write(x_test.shape)
            st.write(y_test.shape)
            
            
            # st.data_editor(x_train)
            # st.data_editor(y_train)
            # model_report=evaluate_model(x_train,y_train,x_test,y_test,models=models)
            search = GridSearchCV(self.pipeline , param_space, n_jobs=-1, cv=5)
            search.fit(x_train, y_train)

            # st.write('Best model:')
            # st.write("///////////////////////")
            # st.write(search.best_params_)
            # st.write("///////////////////////")
            #st.write(search.cv_results_)
            ##Get the best model score from dict
            # best_model_score=max(sorted(model_report.values()))
            best_model_score=search.best_score_
            #st.write(search.cv_results_["mean_test_score"])
            ##Best model name

            # best_model_name=list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]

            # best_model=models[best_model_name]
            best_model=search.best_params_

            if best_model_score<0.0:
                raise CustomException("No best model found")
            logging.info(f"Best Model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # predicted= best_model.predict(x_test)
            # r2_square=r2_score(y_test,predicted)
            return best_model["cls__model_type"],param_space["cls__model_type"],best_model_score,search.cv_results_["mean_test_score"]
        except Exception as e:
            raise CustomException(e,sys)
