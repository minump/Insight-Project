# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:15:43 2020

@author: minum
"""
import sys
import os
import streamlit as st
import argparse
from collections import Counter
from data.data_processing import Preprocessor
from model.XGBoost import XGBoost, Data

st.title('Smartly Financial')


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", action="store", default=os.path.join(os.getcwd(),"data","application_train.zip"), help="dataset path")
    parser.add_argument("--fillna", action="store", default="median", help="Select fill NaNs with: (1)'median' (2)'mean' ")
    parser.add_argument("--sample", action="store", default='SMOTE', 
                        help="Select sampling method from : (1)'undersample' (2)'oversample' (3)'SMOTE' (4) 'SMOTE_undersample' (5) 'None' ")
    args = parser.parse_args()
    
    preprocessor = Preprocessor(args.data)
    #preprocessor.download_data()
    
    df = preprocessor.read_df()
    df, missing_df = preprocessor.missing_data(df)
    #print(missing_df)
    
    preprocessor.data_plot(df, title='Application loans repayed - raw dataset')
    if os.path.basename(args.data)=="application_train.zip":
        df = preprocessor.reform_columns(df)
    
    # select fill NaNs method
    fillnans_selected = st.sidebar.selectbox( 'Select fill NaNs with :', ('median', 'mean'), index=0) # default to median
    
    df = preprocessor.fill_nans(df,fillnans_selected)
    if os.path.basename(args.data)=="application_train.zip":
        num_vars , cat_vars = preprocessor.get_numerical_categorical_var(df)
        df = preprocessor.one_hot_encoding(df, cat_vars)
    
    # select sampling method : under sample majority class, over sample minority class, SMOTE, SMOTE with random under sampling
    sampling_method = st.sidebar.selectbox('Select sampling method from :', 
                                           ('undersample', 'oversample', 'SMOTE with over sampling', 'SMOTE with under sampling', 'None'), index=2)
    
    if sampling_method=='undersample':
        df = preprocessor.undersample(df)
        
    if sampling_method=='oversample':
        df = preprocessor.oversample(df)
    if sampling_method=='SMOTE with over sampling':
        df = preprocessor.SMOTE_oversample(df)
    if sampling_method=='SMOTE with under sampling':
        df = preprocessor.SMOTE_overunder_sample(df)
        
    preprocessor.data_plot(df, title='Application loans repayed - sampled dataset')
    
    
    
    preprocessor.save_train_test_data(df)
    
    
    data = Data()
    
    x_train, y_train, x_val, y_val, x_test, y_test = data.read_data_file()
    data.print_class_ratios(y_train, y_test)
    
    if sampling_method=='None':
        counter_y_train=Counter(y_train)
        estimate= counter_y_train[0]/counter_y_train[1]
        print("ratio of class 0 to 1 :", estimate)
    else:
        estimate=1
    
    # select XGBoost model
    xgboost_selected = st.sidebar.selectbox( 'Select XGBoost model from :', ('XGBClassifier', 'XGBRegressor'), index=0) # default to Classifier
    
    model = XGBoost(xgboost_selected, estimate)
    xgb_clf = model.modelfit(x_train, y_train)
    
    model.plot_feature_importance()
    
    # predicting on val dataset
    val_predicted_labels, val_predicted_proba = model.predict(x_val)
    
    print('Model Performance metrics:')
    print('-' * 30)
    metrics_df = model.get_metrics(y_val, val_predicted_labels)
    print(metrics_df)
    st.write(metrics_df)
    
    print('\nROC plot')
    model.roc_auc_scorer(y_val, val_predicted_proba)
    
    
    
    
if __name__ == "__main__":
    main()
    
    