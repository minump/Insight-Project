# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:51:47 2020

@author: minum
"""

import shap

class Shap:
    
    def __init__(self, model):
        shap.initjs()
        self.explainer = shap.TreeExplainer(model)
        self.shap_values= None
        self.expected_values = None
    
    def explain(self, data):
        self.shap_values = self.explainer.shap_values(data)
        self.expected_value = self.explainer.expected_value
        
    def shap_summaryplot(self, data):
        shap.summary_plot(self.shap_values, data, plot_type='bar')
    
    def shap_dependanceplot(self, data, feature):
        shap.dependence_plot(feature, self.shap_values, data)
