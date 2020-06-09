# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:15:43 2020

@author: minum
"""
import sys
import os
sys.path.append('D:\\Minu\\Insight AI\\Insight-Project\\data')

print(os.getcwd())
print(sys.path)

from data.data_preprocessing import Preprocessor
from model.xgboost_experiment import XGBoost

def main():
    
    preprocessor = Preprocessor()
    preprocessor.download_data()
    df0 = preprocessor.read_data()
    df = preprocessor.reform_columns(df0)
    num_vars , cat_vars = preprocessor.get_numerical_categorical_var(df)
    df_under = preprocessor.undersample(df)
    
    df = preprocessor.normalize(df_under, num_vars)
    df = preprocessor.one_hot_encoding(df, cat_vars)
    
    save_train_test_data(df)
    
    