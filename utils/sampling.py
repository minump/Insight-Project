# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:51:03 2020

@author: minum
"""
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class Sampling:
    
    def __init__(self, input_df):
        #self.input_df = input_df
        self.x = input_df.drop('target', axis=1)
        self.y = input_df['target']
        
    
    # TARGET value 0 means loan is repayed, value 1 means loan is not repayed.
    def undersample(self, random_state=3):
        """
        Undersamples the majority class(target=0) to reach a ratio by default
        equal to 1 between the majority and minority classes
        """
        """
        count_class_0, count_class_1 = self.input_df["target"].value_counts()
        df_class_0 = self.input_df[self.input_df["target"] == 0]
        df_class_1 = self.input_df[self.input_df["target"] == 1]
        df_class_0_under = df_class_0.sample(int(ratio * count_class_1), random_state=random_state)
        df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        return df_train_under
        """
        undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
        x_under , y_under = undersample.fit_resample(self.x, self.y)
        df_under = pd.concat([pd.DataFrame(x_under), pd.DataFrame(y_under, columns=['target'])], axis=1)
        return df_under
        
        
    
    def oversample(self, random_state=3):
        """Oversamples the minority class to reach a ratio by default
            equal to 1 between the majority and mionority classes"""
        """
        count_class_0, count_class_1 = self.input_df["target"].value_counts()
        df_class_0 = self.input_df[self.input_df["target"] == 0]
        df_class_1 = self.input_df[self.input_df["target"] == 1]
        df_class_1_over = df_class_1.sample(int(ratio * count_class_0), replace=True, random_state=random_state)
        df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)
        return df_train_over
        """
        oversample = RandomOverSampler(sampling_strategy=0.5, random_state=random_state)
        x_over , y_over = oversample.fit_resample(self.x, self.y)
        df_over = pd.concat([pd.DataFrame(x_over), pd.DataFrame(y_over, columns=['target'])], axis=1)
        return df_over
        
        
    
    def SMOTE_oversample(self):
        """
        Use SMOTE oversample
        """
        
        oversample = SMOTE()
        x_over, y_over = oversample.fit_resample(self.x, self.y)
        df_smote_over = pd.concat([pd.DataFrame(x_over), pd.DataFrame(y_over, columns=['target'])], axis=1)

        print('SMOTE over-sampling:')
        print(df_smote_over['target'].value_counts())
        #st.write('SMOTE over-sampling:')
        #st.write(df_smote_over['target'].value_counts())

        #df_smote_over['target'].value_counts().plot(kind='bar', title='Count (target)')
        #plt.savefig("df_smote_over.png")
        #st.pyplot()
        
        return df_smote_over
    
    
    def SMOTE_overunder_sample(self):

        oversample = SMOTE(sampling_strategy=0.1)
        undersample = RandomUnderSampler(sampling_strategy=0.5)
        
        x_over, y_over = oversample.fit_resample(self.x, self.y)
        x_under , y_under = undersample.fit_resample(x_over, y_over)
        df_smote_over_under = pd.concat([pd.DataFrame(x_under), pd.DataFrame(y_under, columns=['target'])], axis=1)
        
        print('SMOTE + random under sampling:')
        print(df_smote_over_under['target'].value_counts())
        #st.write('SMOTE + random under sampling:')
        #st.write(df_smote_over_under['target'].value_counts())

        #df_smote_over_under['target'].value_counts().plot(kind='bar', title='Count (target)')
        #st.pyplot()
        #plt.savefig("df_smote_over_under.png")
        
        return df_smote_over_under