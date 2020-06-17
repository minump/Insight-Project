# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:15:43 2020

@author: minum
"""
import sys
import os
import streamlit as st
import argparse

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from collections import Counter
from data.data_processing import Preprocessor
from data.sampling import Sampling
from data.save_data import SaveData
from model.XGBoost import XGBoost, Data
from model.Shap import Shap
import joblib 
import pickle

st.title('Smartly Financial')


def preprocessing(config):
    """
    Preprocessing data
    """
    
    preprocessor = Preprocessor(config)
    #preprocessor.download_data()
    
    df = preprocessor.read_df()
    df, missing_df = preprocessor.missing_data(df)
    #print(missing_df)
    st.write("Percentage of missing data")
    st.dataframe(missing_df)
    
    # select fill NaNs method
    fillnans_selected = st.sidebar.selectbox( 'Select fill NaNs with :', ('median', 'mean'), index=0) # default to median
    df = preprocessor.fill_nans(df,fillnans_selected)
    
    
    if os.path.basename(config)=="application_train.zip":
        df = preprocessor.reform_columns(df)
    
    
    if os.path.basename(config)=="application_train.zip":
        num_vars , cat_vars = preprocessor.get_numerical_categorical_var(df)
        df = preprocessor.one_hot_encoding(df, cat_vars)
        
    return df


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
    

    
def data_plot(df0, title):
    # target value 0 means loan is repayed, value 1 means loan is not repayed.

    temp = df0["target"].value_counts()
    df0_target = pd.DataFrame({'labels': temp.index, 'values': temp.values })
    plt.figure(figsize = (10,10))
    plt.title(title)
    sns.set_color_codes("pastel")
    sns.barplot(x = 'labels', y="values", data=df0_target)
    locs, labels = plt.xticks()
    st.pyplot()
    print(title," ratio of 0 to 1 in target column ")
    print(df0['target'].value_counts(normalize=True) * 100)
    st.write(title, " Target value counts")
    st.dataframe(df0['target'].value_counts(normalize=True) * 100)
        
    

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", action="store", default=os.path.join(os.getcwd(),"data","sample_data.zip"), help="dataset path")
    parser.add_argument("--fillna", action="store", default="median", help="Select fill NaNs with: (1)'median' (2)'mean' ")
    parser.add_argument("--sample", action="store", default='SMOTE', 
                        help="Select sampling method from : (1)'undersample' (2)'oversample' (3)'SMOTE' (4) 'SMOTE_undersample' (5) 'None' ")
    args = parser.parse_args()
    
    df = preprocessing(args.data)
    
    data_plot(df, title='Application loans repayed - raw dataset')
    
    # select sampling method : under sample majority class, over sample minority class, SMOTE, SMOTE with random under sampling
    sampling_method = st.sidebar.selectbox('Select sampling method from :', 
                                           ('undersample', 'oversample', 'SMOTE with over sampling', 'SMOTE with under sampling', 'None'), index=4)
    
    df = sample(df, sampling_method)
    
    data_plot(df, title='Application loans repayed - sampled dataset')
    
    savingData = SaveData(df)
    
    savingData.save_train_test_data()
    
    data = Data()
    
    x_train, y_train, x_val, y_val, x_test, y_test = data.read_data_file()
    #print(x_train.head())
    #print(x_val.head())
    #print(x_test.head())
    
    data.print_class_ratios(y_train, y_test)
    
    if sampling_method=='None':
        counter_y_train=Counter(y_train)
        estimate= counter_y_train[0]/counter_y_train[1]
        print("ratio of class 0 to 1 :", estimate)
        st.write("Scale factor for classes ", estimate)
    else:
        estimate=1  
    
    
    # select XGBoost model
    xgboost_selected = st.sidebar.selectbox( 'Select XGBoost model from :', ('XGBClassifier', 'XGBRegressor'), index=0) # default to Classifier
    
    model = XGBoost(xgboost_selected, estimate)
    xgb_clf = model.modelfit(x_train, y_train)
    
    model.plot_feature_importance()
    
    # predicting on val dataset
    val_predicted_labels, val_predicted_proba = model.predict(x_val)
    
    print('Model Performance metrics on validation data:')
    print('-' * 30)
    metrics_df = model.get_metrics(y_val, val_predicted_labels)
    print(metrics_df)
    
    print('\nROC plot')
    st.write('ROC score on validation data')
    model.roc_auc_scorer(y_val, val_predicted_proba)
    
    
    
    # predict on user test data
    user_testdata_index = st.sidebar.selectbox("Select from user test data :", x_test.index)
    user_testdata_predicted_label, user_testdata_predicted_proba = model.predict(x_test.loc[[user_testdata_index]])
    if user_testdata_predicted_label==1.0:
        st.write("The user with user_id ",user_testdata_index, " predicted : Defaulted")
    else:
        st.write("The user with user_id ",user_testdata_index, " predicted : Paid")
        
    
    
    # explain with test data
    
    explain_button = st.sidebar.button('Explain with test data')
    
    if explain_button:
        shap = Shap(xgb_clf)
        shap.explain(x_test)
        shap.shap_summaryplot(x_test)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        shap.shap_dependanceplot(x_test, 'f_104')
        #st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        
    

        
        
    
    
    
    
if __name__ == "__main__":
    main()
    
    