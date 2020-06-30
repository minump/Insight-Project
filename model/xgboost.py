#!/usr/bin/env python
# coding: utf-8
"""
Use XGBoost model and print the performance metrics
"""


import os
import os.path
from os import path
from importlib import reload
import streamlit as st
import joblib 

import pandas as pd
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import time
import numpy as np

from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import xgboost


class XGBoost:
    """
    XGBoost model with CV hyper parameter tuning.
    Return compiled model
    Fit model
    Print results
    """
    
    def __init__(self, scale_factor, filename, sampling):
        
        self.dataset=filename
        self.sampling=sampling
        
        self.xgb_model = xgboost.XGBClassifier( max_depth=5, min_child_weight=1, gamma=0,
                                               subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                               nthread=4, scale_pos_weight=scale_factor, seed=27, tree_method='gpu_hist')
                    
        self.title='XGBoostClassifier'
            
        
        param_grid ={'learning_rate':[0.01, 0.05],'n_estimators':[100, 300] } 
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        self.clf = GridSearchCV( estimator=self.xgb_model, n_jobs=2, param_grid=param_grid, cv=cv,
                                return_train_score=True, scoring='roc_auc',verbose=3)
        self.clf_model=None
            

    def modelfit(self, x_train, y_train):
        """
        Trains the model and returns the best estimated model. 
        Prints training performance
        """
        if path.exists(os.path.join(os.getcwd(),"model","saved_models",self.dataset+'_'+self.sampling+'_xgb_clf_model.pkl')):
            st.write("Loading saved XGB model")
            print("Loading saved XGB model")
            # Load the model from the file 
            self.clf_model = joblib.load(os.path.join(os.getcwd(),"model","saved_models",self.dataset+'_'+self.sampling+'_xgb_clf_model.pkl'))
            #test_predict_labels = self.clf_model.predict(x_train.iloc[0:,])
            
        else:
            start_time = time.time()
            st.write('Training model', self.title)
            print("Training model ", self.title)
    
            #Fit the algorithm on the data
            self.clf.fit(x_train, y_train,eval_metric='auc')
        
            print ('Model trained in seconds ',format(time.time() - start_time))
            st.write('Model trained in seconds ',format(time.time() - start_time))
            #Predict training set:
            dtrain_predictions = self.clf.predict(x_train)
            dtrain_predprob = self.clf.predict_proba(x_train)[:,1]
        
            #Print model report:
            print ("Train Average Precision Score : " , metrics.average_precision_score(y_train, dtrain_predictions))
            st.write("Train Average Precision Score : " , metrics.average_precision_score(y_train, dtrain_predictions))
        
            self.clf_model = self.clf.best_estimator_
        
            joblib.dump(self.clf_model, os.path.join(os.getcwd(),"model","saved_models",self.dataset+'_'+self.sampling+'_xgb_clf_model.pkl') )
            
        #self.display_model_params()
        
        
        return self.clf_model
    

    def plot_feature_importance(self):
        feat_imp = pd.Series(self.clf_model.feature_importances_).sort_values(ascending=False)[:20]
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.savefig("feature_importance.png")
        st.pyplot()
        


    #clf = XGBClassifier(n_estimators=1000, objective='binary:logistic', gamma=0.1, subsample=0.5, scale_pos_weight=ratio )



    """
    base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0.1, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.300000012, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=nan, monotone_constraints='()',
       n_estimators=26, n_jobs=0, num_parallel_tree=1,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=11.329374028968076, subsample=0.5,
       tree_method='exact', validate_parameters=1, verbosity=None)
    """
    
    def display_model_params(self):
        print('Best estimator:')
        print(self.clf_model)
        st.write('Best estimator:')
        st.write(self.clf_model)
        
        print('Best normalized gini score')
        print(self.clf_model.best_score_ * 2 - 1)
        st.write('Best normalized gini score ', self.clf_model.best_score_ * 2 - 1)
        
        print('Best hyperparameters:')
        print(self.clf_model.best_params_)
        st.write('Best hyperparameters:')
        st.write(self.clf_model.best_params_)
        results = pd.DataFrame(self.clf_model.cv_results_)
        results.to_csv(os.path.join(os.getcwd(),"model","saved_models",self.dataset+'_'+self.sampling+'_xgb_gridsearch_results.csv'), index=False)



    def get_model(self):
        return self.clf_model
    
    def get_model_title(self):
        return self.title







