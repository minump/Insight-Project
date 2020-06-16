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
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data import read_data
from data import df_one_hot_encode
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
 

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
        
        st.write('Number of samples :', df0.shape[0])
        st.write('Number of features :', df0.shape[1])
        return df0


    # find percentage of missing data
    def missing_data(self, df):
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        #print(empty_cols)
        df.drop(empty_cols, axis=1, inplace=True)
        df=df.loc[:, (df != 0).any(axis=0)]  # remove columns with all zeros
        #print(df.shape)   # (62641, 230)  -- 2 columns have all zeros.
        
        total = df.isnull().sum().sort_values(ascending = False)
        percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
        st.bar_chart(percent[:30])
        #percent.plot.barh()
        #st.pyplot()
        return df, pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



    #missing_data(df0).head(20)


    def data_plot(self, df0, title):
        # target value 0 means loan is repayed, value 1 means loan is not repayed.

        temp = df0["target"].value_counts()
        df0_target = pd.DataFrame({'labels': temp.index, 'values': temp.values })
        plt.figure(figsize = (6,6))
        plt.title(title)
        sns.set_color_codes("pastel")
        sns.barplot(x = 'labels', y="values", data=df0_target)
        locs, labels = plt.xticks()
        st.pyplot()
        print(title," ratio of 0 to 1 in target column ")
        print(df0['target'].value_counts(normalize=True) * 100)
        


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


    # TARGET value 0 means loan is repayed, value 1 means loan is not repayed.
    def undersample(self, input_df, ratio=1.0, random_state=3):
        """
        Undersamples the majority class(target=0) to reach a ratio by default
        equal to 1 between the majority and minority classes
        """
        count_class_0, count_class_1 = input_df["target"].value_counts()
        df_class_0 = input_df[input_df["target"] == 0]
        df_class_1 = input_df[input_df["target"] == 1]
        df_class_0_under = df_class_0.sample(int(ratio * count_class_1), random_state=random_state)
        df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        return df_train_under
    
    def oversample(self, input_df, ratio=1.0, random_state=3):
        """Oversamples the minority class to reach a ratio by default
            equal to 1 between the majority and mionority classes"""
        count_class_0, count_class_1 = input_df["target"].value_counts()
        df_class_0 = input_df[input_df["target"] == 0]
        df_class_1 = input_df[input_df["target"] == 1]
        df_class_1_over = df_class_1.sample(int(ratio * count_class_0), replace=True, random_state=random_state)
        df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)
        return df_train_over
    
    def SMOTE_oversample(self, input_df):
        
        x = input_df.drop('target', axis=1)
        y = input_df['target']
        
        oversample = SMOTE()
        x_over, y_over = oversample.fit_resample(x, y)
        df_smote_over = pd.concat([pd.DataFrame(x_over), pd.DataFrame(y_over, columns=['target'])], axis=1)

        print('SMOTE over-sampling:')
        print(df_smote_over['target'].value_counts())
        st.write('SMOTE over-sampling:')
        st.write(df_smote_over['target'].value_counts())

        df_smote_over['target'].value_counts().plot(kind='bar', title='Count (target)')
        plt.savefig("df_smote_over.png")
        st.pyplot()
        
        return df_smote_over
    
    
    def SMOTE_overunder_sample(self, input_df):
        x = input_df.drop('target', axis=1)
        y = input_df['target']
        oversample = SMOTE(sampling_strategy=0.1)
        undersample = RandomUnderSampler(sampling_strategy=0.5)
        
        x_over, y_over = oversample.fit_resample(x, y)
        x_under , y_under = undersample.fit_resample(x_over, y_over)
        df_smote_over_under = pd.concat([pd.DataFrame(x_under), pd.DataFrame(y_under, columns=['target'])], axis=1)
        
        print('SMOTE + random under sampling:')
        print(df_smote_over_under['target'].value_counts())
        st.write('SMOTE + random under sampling:')
        st.write(df_smote_over_under['target'].value_counts())

        df_smote_over_under['target'].value_counts().plot(kind='bar', title='Count (target)')
        st.pyplot()
        plt.savefig("df_smote_over_under.png")
        
        return df_smote_over_under
        
        
        

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
    

    def save_train_test_data(self, df):
        # split into train val and test
        train, test = train_test_split( df, train_size = 0.8, test_size = 0.2 )
        #print(train.shape, test.shape) #(39720, 175) (9930, 175)
        train = shuffle(train)
        partial_train = train[:31776]
        val=train[31776:]
        #print(partial_train.shape, val.shape, test.shape) # (31776, 175) (7944, 175) (9930, 175)

        y_train = partial_train['target']
        y_val = val['target']
        y_test  = test['target']

        x_train = partial_train.loc[ : , partial_train.columns != 'target']
        x_val =  val.loc[:,val.columns!='target']
        x_test  = test .loc[ : , test.columns != 'target']
        #print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)    
        #(31776, 174) (7944, 174) (9930, 174) (31776,) (7944,) (9930,)


        x_train.to_hdf('data/x_train.h5', key='df', mode='w')
        y_train.to_hdf('data/y_train.h5', key='df', mode='w')
        x_val.to_hdf('data/x_val.h5', key='df', mode='w')
        y_val.to_hdf('data/y_val.h5', key='df', mode='w')
        x_test.to_hdf('data/x_test.h5', key='df', mode='w')
        y_test.to_hdf('data/y_test.h5', key='df', mode='w')
