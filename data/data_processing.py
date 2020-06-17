#!/usr/bin/env python
# coding: utf-8
"""
Data : Home credit application loan dataset : application_train.csv
Preprocess the data and save h5 files
"""

import os
import streamlit as st
import zipfile,fnmatch
import pandas as pd
from sklearn import preprocessing

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data import read_data
from data import df_one_hot_encode


 

class Preprocessor:
    
    def __init__(self,path):
        self.path=path
        self.filename=os.path.basename(path)   # application_train.zip
        self.dir=os.path.dirname(path)
    
    
    # TARGET value 0 means loan is repayed, value 1 means loan is not repayed.
    def read_df(self):
        
        data = read_data.ReadData(self.path)
        data.unzip_data()
        df0 = data.read_data()
        
        df0.columns = [ col.lower() for col in df0.columns ]
        #desc = df.describe().transpose()
        #print(desc)
        #print(df.shape) # (307511, 122)
        if self.filename=="application_train.zip":
            df0 = df0.set_index( 'sk_id_curr')  # no need for column ID in data analysis.

            df0.rename( columns={ "days_birth" : "age",
                                 "name_education_type" : "education",
                                 "name_housing_type" : "housing",
                                 "name_income_type"  : "income",
                                 "name_family_status" : "fam_status",
                                 "code_gender" : "gender"}, inplace=True)
            df0['education'] = ( df0['education'].replace('Secondary / secondary special', 'Secondary') .replace( 'Higher education', 'Higher') )
        
        if self.filename=="sample_data.zip":
            df0 = df0.set_index('user_id')
        
        st.write('Number of samples in raw data:', df0.shape[0])
        st.write('Number of features in raw data :', df0.shape[1])
        return df0


    # find percentage of missing data
    def missing_data(self, df):
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        #print(empty_cols)
        if empty_cols:
            df.drop(empty_cols, axis=1, inplace=True)
        
        df=df.loc[:, (df != 0).any(axis=0)]  # remove columns with all zeros
        #print(df.shape)   # (62641, 230)  -- 2 columns have all zeros.
        
        total = df.isnull().sum().sort_values(ascending = False)
        percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
        #st.bar_chart(percent[:30])
        #percent.plot.barh()
        #st.pyplot()
        return df, pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        


    def reform_columns(self, df0):
        """
        delete some columns; transform some columns as boolean; fill NaNs with median
        """
        df =df0.copy()
        del df['organization_type'] # dropped because of too many values
        del df['ext_source_1']
        del df['ext_source_2']
        del df['ext_source_3']
        df['flag_own_car'] = df0['flag_own_car'] == 'Y'
        df['flag_own_realty'] = df0['flag_own_realty'] == 'Y'
        del df['amt_req_credit_bureau_hour']
        del df['amt_req_credit_bureau_day']
        del df['amt_req_credit_bureau_week']
        del df['amt_req_credit_bureau_mon']
        del df['amt_req_credit_bureau_qrt']
        del df['amt_req_credit_bureau_year']

        # Convert flag columns to bool
        for col in df.columns :
            if col.startswith( 'flag_' ) or col.startswith( 'reg_') : 
                df[col] = (df[col] == 1)
        #print( df[col].value_counts() )
        return df
    
    def fill_nans(self, df, option):
        # For float64 columns impute  NaNs with median or mean

        for col in df.select_dtypes('float64').columns : 
            if df[col].isnull().sum() > 0 :  
                if option=='median':
                    median = df[col].median() 
                    df[col] = df[col].fillna(  median )
                if option=='mean':
                    mean=df[col].mean()
                    df[col] = df[col].fillna(mean)
        return df

    
    def get_numerical_categorical_var(self, df):
        # get all numerical columns names
        num_vars = list(df.select_dtypes(include=[np.number]).columns.values)

        num_vars.remove('target')
        #print(num_vars)
        cat_vars = df.dtypes[ df.dtypes == 'object' ]
        #print("cat_vars")
        #print(cat_vars)
        return num_vars, cat_vars
        

    def normalize(self, df, columns):
        min_max_scaler = preprocessing.MinMaxScaler()
        df[columns] = min_max_scaler.fit_transform(df[columns])
        return df
    
    
    def one_hot_encoding(self, df, cat_vars):
        if cat_vars is not None:
            oh_enc = df_one_hot_encode.DfOneHotEncoder( cat_vars.index )
            oh_enc.fit( df )
            df = oh_enc.transform( df, drop_old=True )
        return df
    

    
