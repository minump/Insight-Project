# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:42:18 2020

@author: minum
"""


import os
import zipfile
import streamlit as st
import pandas as pd

@st.cache
class ReadData:
    """
    unzip and read data into df. Returns the df.
    """
    def __init__(self,path):
        self.path=path
        self.filename=os.path.basename(path)   # application_train.zip
        self.dir=os.path.dirname(path)
        
    def unzip_data(self):
        zfile = zipfile.ZipFile(self.path)
        zfile.extractall(self.dir)
        zfile.close()
    
    def read_data(self):
        st.write('Loading data from', self.filename)
        print("loading data from ", self.filename)
        csvfile = os.path.splitext(self.filename)[0] + '.csv' # application_train
        csvfilename = os.path.join(self.dir,csvfile)
        df0 = pd.read_csv(csvfilename)
        #print(df0.columns.values)
        return df0
