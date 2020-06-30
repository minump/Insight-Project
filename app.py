# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:15:43 2020

@author: minum
"""

import os
import streamlit as st
import argparse
from importlib import reload

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from collections import Counter
from utils.data_processing import Preprocessor
from utils.sampling import Sampling
from utils.split_data import SplitData
from model.XGBoost import XGBoost
from model import performance_metrics
from model.LogisticRegression import LR
from explain.Shap import Shap
import joblib 

import sys
#sys.path.append('D:\\Minu\\Insight AI\\Insight-Project\\')
#sys.path.append('D:\\Minu\\Insight AI\\Insight-Project\\src')


st.title('Smartly Financial')

@st.cache
def preprocessing(data_path, fill_option):
    """
    Preprocessing data
    """
    preprocessor = Preprocessor(data_path)
    df, filename = preprocessor.read_df()
    df, missing_df = preprocessor.missing_data(df)
    df = preprocessor.fill_nans(df,fill_option)
    
    if os.path.basename(data_path)=="application_train.zip":
        df = preprocessor.reform_columns(df)
    
    num_vars , cat_vars = preprocessor.get_numerical_categorical_var(df)
    
    if os.path.basename(data_path)=="application_train.zip":
        df = preprocessor.one_hot_encoding(df, cat_vars)
    
    df = preprocessor.normalize(df, num_vars)
    
    return df, filename


@st.cache()
def sample(df, sampling_method):
    
    sample = Sampling(df)
    
    if sampling_method=='undersample':
        df = sample.undersample()
        
    if sampling_method=='oversample':
        df = sample.oversample()
    if sampling_method=='SMOTE with over sampling':
        df = sample.SMOTE_oversample()
    if sampling_method=='SMOTE with under sampling':
        df = sample.SMOTE_overunder_sample()
    if sampling_method=='None':
        pass
    
    return df

def build_model(option, estimate, filename, sampling, x_train, y_train):
    
    st.write('Build model')
    print('Build model')
    
    if option=='XGBClassifier':
        model = XGBoost(estimate, filename, sampling)
    if option=='Logistic Regression':
        model = LR(filename, sampling)
        
    clf = model.modelfit(x_train, y_train)
    
    #model.plot_feature_importance()
    return model.get_model(), model.get_model_title()


def predictions(model, x ):
    predicted_labels = model.predict(x)
    predicted_proba = model.predict_proba(x)[:,1]
    return predicted_labels, predicted_proba    
    
    
    
def data_plot(df0, title):
    # target value 0 means loan is repayed, value 1 means loan is not repayed.
    
    #temp = df0["target"].value_counts()
    #df0_target = pd.DataFrame({'labels': temp.index, 'values': temp.values })
    print(title," ratio of 0 to 1 in target column ")
    print(df0['target'].value_counts(normalize=True) * 100)
    st.write(title, " Target percentage counts")
    st.dataframe(df0['target'].value_counts(normalize=True) * 100)


def print_class_ratios(y_train, y_test):
    
    ratio = (y_train == 0).sum()/ (y_train == 1).sum()
    print("ratio of 0 to 1 in target column ", ratio)
    print("Postive examples in train set: {}".format(np.sum(y_train==0)))
    print("Negative examples in train set: {}".format(np.sum(y_train==1)))

    print("Postive examples in test set: {}".format(np.sum(y_test==0)))
    print("Negative examples in test set: {}".format(np.sum(y_test==1)))        
    

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", action="store", default=os.path.join(os.getcwd(),"data","application_train.zip"), help="dataset path")
    parser.add_argument("--fillna", action="store", default="median", help="Select fill NaNs with: (1)'median' (2)'mean' ")
    parser.add_argument("--sample", action="store", default='SMOTE', 
                        help="Select sampling method from : (1)'undersample' (2)'oversample' (3)'SMOTE' (4) 'SMOTE_undersample' (5) 'None' ")
    parser.add_argument("--model", action="store", default="XGBClassifier", help="Select model from: (1)'Logistic Regression' (2)'XGBClassifier' ")
    args = parser.parse_args()
    
    st.write('Loading data from', os.path.basename(args.data))
    df, filename = preprocessing(args.data, args.fillna)
    st.write('Number of samples in raw data:', df.shape[0])
    st.write('Number of features in raw data :', df.shape[1])
    
    # define the target labels for 2 different datasets
    classes = df['target'].unique()
        
    
    data_plot(df, title='Raw dataset')
    
    # select sampling method : under sample majority class, over sample minority class, SMOTE, SMOTE with random under sampling
    sampling_method = st.sidebar.selectbox('Select sampling method from :', ('undersample', 'oversample', 'SMOTE', 'SMOTE with under sampling', 'None'), index=3)
    
    df = sample(df, sampling_method)
    
    data_plot(df, title='Sampled dataset')
    
    splitingData = SplitData(df)
    
    x_train, y_train, x_val, y_val, x_test, y_test = splitingData.split_train_test_data()
    print_class_ratios(y_train, y_test)
    
    if sampling_method=='None':
        counter_y_train=Counter(y_train)
        estimate= counter_y_train[0]/counter_y_train[1]
        print("ratio of class 0 to 1 :", estimate)
        st.write("Scale factor for classes ", estimate)
    else:
        estimate=1  
    
    
    # select model
    model_selected = st.sidebar.selectbox( 'Select model from :', ('Logistic Regression', 'XGBClassifier'), index=1) # default to Classifier
    
    
    model, title = build_model(model_selected, estimate, filename, sampling_method, x_train, y_train)
    
    # predicting on test dataset
    test_predicted_labels, test_predicted_proba = predictions(model, x_test)
    
    
    performance_metrics.display_model_performance_metrics(model, title, x_test, y_test, 
                                                          test_predicted_labels, test_predicted_proba, classes=classes)
    
    
    
    
    # predict on user test data
    user_testdata_index = st.sidebar.selectbox("Select from user test data :", x_test.index)
    user_testdata_predicted_label, user_testdata_predicted_proba = predictions(model, x_test.loc[[user_testdata_index]])
    if user_testdata_predicted_label==1.0 or user_testdata_predicted_label==1:
        st.write("The user with user_id ",user_testdata_index, " predicted : Defaulted")
        print("The user with user_id ",user_testdata_index, " predicted : Defaulted")
    else:
        st.write("The user with user_id ",user_testdata_index, " predicted : Paid")
        print("The user with user_id ",user_testdata_index, " predicted : Paid")
        
    
    
    # explain with test data
    if model_selected=='XGBClassifier':
        st.write('Explaining XGBClassifier with SHAP tree explainer')
        xgb_clf = joblib.load(os.path.join(os.getcwd(),"model","saved_models",str(filename)+'_'+str(sampling_method)+'_xgb_clf_model.pkl'))
    
        shap = Shap(xgb_clf)
        shap.explain(x_test)
    
        shap.shap_summaryplot(x_test)
        #feature_set =['f_2', 'f_7','f_0', 'f_104']
        shap.shap_dependanceplot(x_test)
        
    

        
        
    
    
    
    
if __name__ == "__main__":
    main()
    
    
