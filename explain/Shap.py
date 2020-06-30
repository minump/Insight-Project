# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:51:47 2020

@author: minum
"""

import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

class Shap:
    
    def __init__(self, model):
        shap.initjs()
        self.explainer = shap.TreeExplainer(model)
        self.shap_values= None
        self.expected_values = None
        self.feature_importance = None
    
    def explain(self, data):
        self.shap_values = self.explainer.shap_values(data)
        self.expected_value = self.explainer.expected_value
        
        vals= np.abs(self.shap_values).mean(0)

        self.feature_importance = pd.DataFrame(list(zip(data.columns, vals)), columns=['col_name','feature_importance_vals'])
        self.feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        st.write('Feature importance for test data :')
        st.dataframe(self.feature_importance)
        
        
    
    def shap_summaryplot(self, data):
        f1 = plt.figure()
        shap.summary_plot(self.shap_values, data, plot_type='bar')
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        f1.savefig("summary_plot_bartype.png", bbox_inches='tight', dpi=600)
        f2 = plt.figure()
        shap.summary_plot(self.shap_values, data)
        st.pyplot(dpi=300, pad_inches=0)
        f2.savefig("summary_plot.png", bbox_inches='tight', dpi=600)
    
        
    
    def shap_dependanceplot(self, data):
        # showing dependance plot for top 4 features 
        feature_set=self.feature_importance['col_name'][:5]
        for f in feature_set:
            shap.dependence_plot(f, self.shap_values, data)
            st.pyplot()
        
    def shap_summary_correlation_plot(self, df):
        shap_v = pd.DataFrame(self.shap_values)
        feature_list = df[self.feature_importance[:20]].columns
        shap_v.columns = feature_list
        df_v = df.copy()
    
        # Determine the correlation in order to plot with different colors
        corr_list = list()
        for i in feature_list:
            b = np.corrcoef(shap_v[i],df_v[i])[1][0]
            corr_list.append(b)
        corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
        # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
        corr_df.columns  = ['Variable','Corr']
        corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
        # Plot it
        f = plt.figure()
        shap_abs = np.abs(shap_v)
        k=pd.DataFrame(shap_abs.mean()).reset_index()
        k.columns = ['Variable','SHAP_abs']
        k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
        k2 = k2.sort_values(by='SHAP_abs',ascending = True)
        colorlist = k2['Sign']
        ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
        ax.set_xlabel("SHAP Value (Red = Positive Impact)")
        st.pyplot()
        f.savefig("summary_correlation_plot.png")
        
