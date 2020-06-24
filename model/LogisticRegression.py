# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:33:52 2020

@author: minum
"""

import os
import os.path
from os import path
from importlib import reload
import streamlit as st
import joblib 

import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import time
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class LR:
    """
    Defines Logistic Regression model from sklearn library.
    Trains and returns the fitted model
    Predicts on test and validation dataset
    """
    
    def __init__(self, filename, sampling):
        self.dataset = filename
        self.sampling=sampling
        
        self.lr_model = LogisticRegression(solver='sag')  # sag solver for large datasets
        self.title = 'LogisticRegression'
    
    def modelfit(self, x_train, y_train):
        """
        Trains the model and returns the best estimated model. 
        Prints training performance
        """
        if path.exists(os.path.join(os.getcwd(),"model","saved_models",self.dataset+'_'+self.sampling+'_lr_clf_model.pkl')):
            st.write("Loading saved LR model")
            print("Loading saved LR model")
            # Load the model from the file 
            self.lr_model = joblib.load(os.path.join(os.getcwd(),"model","saved_models",self.dataset+'_'+self.sampling+'_lr_clf_model.pkl'))
            
        else:
            start_time = time.time()
            st.write('Training model', self.title)
            print("Training model ", self.title)
    
            #Fit the algorithm on the data
            self.lr_model.fit(x_train, y_train)
        
            print ('Model trained in seconds ',format(time.time() - start_time))
            st.write('Model trained in seconds ',format(time.time() - start_time))
            #Predict training set:
            dtrain_predictions = self.lr_model.predict(x_train)
            dtrain_predprob = self.lr_model.predict_proba(x_train)[:,1]
        
            #Print model report on training set:
            print ("Train Average Precision Score : " , metrics.average_precision_score(y_train, dtrain_predictions))
            st.write("Train Average Precision Score : " , metrics.average_precision_score(y_train, dtrain_predictions))
        
        
            joblib.dump(self.lr_model, os.path.join(os.getcwd(),"model","saved_models",self.dataset+'_'+self.sampling+'_lr_clf_model.pkl') )
        
        
        return self.lr_model
    
    def predictions(self, x ):
        predicted_labels = self.lr_model.predict(x)
        predicted_proba = self.lr_model.predict_proba(x)[:,1]
        return predicted_labels, predicted_proba
    
    def get_model(self):
        return self.lr_model
    
    def get_model_title(self):
        return self.title