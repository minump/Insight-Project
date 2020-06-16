#!/usr/bin/env python
# coding: utf-8
"""
Use XGBoost model and print the performance metrics
"""


from collections import OrderedDict, Counter
from itertools import product
import random

import os
from importlib import reload
import streamlit as st

import pandas as pd
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import time
import numpy as np

from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
import xgboost



#import performance_metrics


class Data:
    def __init__(self):
        pass

    def read_data_file(self):
        # read h5 file from data folder
        x_train = pd.read_hdf('data/x_train.h5')
        y_train = pd.read_hdf('data/y_train.h5')
        x_val = pd.read_hdf('data/x_val.h5') 
        y_val = pd.read_hdf('data/y_val.h5')
        x_test = pd.read_hdf('data/x_test.h5')
        y_test = pd.read_hdf('data/y_test.h5')
        #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
        #(31776, 174) (31776,) (7944, 174) (7944,) (9930, 174) (9930,)
        return x_train, y_train, x_val, y_val, x_test, y_test


    def print_class_ratios(self, y_train, y_test):

        ratio = (y_train == 0).sum()/ (y_train == 1).sum()
        print("ratio of 0 to 1 in target column ", ratio)
        print("Postive examples in train set: {}".format(np.sum(y_train==0)))
        print("Negative examples in train set: {}".format(np.sum(y_train==1)))

        print("Postive examples in test set: {}".format(np.sum(y_test==0)))
        print("Negative examples in test set: {}".format(np.sum(y_test==1)))



class XGBoost:
    """
    XGBoost model with CV hyper parameter tuning.
    Return compiled model
    Fit model
    Print results
    """
    
    def __init__(self, model_option, scale_factor):
        if model_option=='XGBClassifier':
            self.xgb_model = xgboost.XGBClassifier( max_depth=5, min_child_weight=1, gamma=0,
                                           subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                           nthread=4, scale_pos_weight=scale_factor, seed=27, tree_method='gpu_hist')
                    
            self.title='XGBoostClassifier'
            
        if model_option=='XGBRegressor':
            self.xgb_model = xgboost.XGBRegressor(tree_method='gpu_hist')
            self.title='XGBoostRegressor'
        
        param_grid ={'learning_rate':[0.01, 0.05],'n_estimators':[100, 300] } 
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        self.clf = GridSearchCV( estimator=self.xgb_model, n_jobs=1, param_grid=param_grid, cv=cv,
                                return_train_score=True, scoring='roc_auc',verbose=3)
            
        

    def modelfit(self, x_train, y_train):
        
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
        print ("Train Accuracy : " , metrics.accuracy_score(y_train, dtrain_predictions))
        st.write("Train Accuracy : " , metrics.accuracy_score(y_train, dtrain_predictions))
        print ("AUC Score (Train):", metrics.roc_auc_score(y_train, dtrain_predprob))
        st.write("AUC Score (Train):", metrics.roc_auc_score(y_train, dtrain_predprob))
        
        return self.clf
    
    
    def plot_feature_importance(self):
        feat_imp = pd.Series(self.clf.best_estimator_.feature_importances_).sort_values(ascending=False)[:20]
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
        print(self.clf.best_estimator_)
        print('Best normalized gini score')
        print(self.clf.best_score_ * 2 - 1)
        print('Best hyperparameters:')
        print(self.clf.best_params_)
        results = pd.DataFrame(self.clf.cv_results_)
        results.to_csv('xgb-grid-search-results-01.csv', index=False)


    def predict(self, x ):
        predicted_labels = self.clf.predict(x)
        predicted_proba = self.clf.predict_proba(x)[:,1]
        return predicted_labels, predicted_proba



    def get_metrics(self, true_labels, predicted_labels):
        accuracy = np.round(metrics.accuracy_score(true_labels, predicted_labels), 4)
        prec = np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 4)
        recall = np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 4)
        f1 = np.round( metrics.f1_score(true_labels, predicted_labels, average='weighted'), 4)

        df = pd.DataFrame([[accuracy, prec, recall, f1]], index=['performance'], columns=["accuracy", "precision", "recall", "f1_score"])
        print(metrics.classification_report(true_labels, predicted_labels, labels=[1,0]))
        return df


    # get AUC score for both train and test data sets

    def roc_auc_scorer(self, y, y_pred ) :
        roc_score = metrics.roc_auc_score( y, y_pred )
        print("roc_score ",roc_score)
        st.write("roc_score ",roc_score)
        fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    
        # plot the roc curve for the model
        pyplot.figure()
        pyplot.plot(fpr, tpr, marker='.', label='XGBoost')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        st.pyplot()





