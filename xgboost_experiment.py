#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Try XGBoost model on Home Credit Risk Data from Kaggle competition.
# Data - only application data(train and test). Contains info about loan application. 
# Training data has target column with 0 loan repaid and 1 loan not repaid.
# Grid search hyper parameters for XGBoost model

from collections import OrderedDict
from itertools import product
import random

import os
from importlib import reload

import pandas as pd
import xgboost as xgb
import numpy as np

#import df_one_hot_encode

#from memoize import Memoizer
#from inst_func_eval import InstFunEvaluator


# In[13]:


#DATA_DIR = r"D:\Minu\Insight/ AI\Lending/ Decisions\XGBoostData/"
app_fn = "application_train.csv"

df0 = pd.read_csv( app_fn )


# In[15]:


df0.columns = [ col.lower() for col in df0.columns ]
desc = df0.describe().transpose()
print(desc)
print(df0.shape) # (307511, 122)


# In[16]:


df0.columns


# In[17]:


df0 = df0.set_index( 'sk_id_curr')  # no need for column ID in data analysis.

df0.rename( columns={ "days_birth" : "age",
                      "name_education_type" : "education",
                      "name_housing_type" : "housing",
                      "name_income_type"  : "income",
                      "name_family_status" : "fam_status",
                      "code_gender" : "gender"}, inplace=True)


# In[18]:


df0['ext_source_1'][:5]


# In[23]:


df0[df0.isnull().any(1)]
df0.columns[df0.isnull().any()]


# In[24]:


df0['education'] = ( df0['education'].replace('Secondary / secondary special', 'Secondary')
                                     .replace( 'Higher education', 'Higher') )
df0['flag_own_car'][:5]


# In[43]:


df0['ext_source_1'].isnull().sum()


# In[25]:


df1 = df0.copy()
del df1['organization_type'] # dropped because of too many values
del df1['ext_source_1']
del df1['ext_source_2']
del df1['ext_source_3']
df1['flag_own_car'] = df0['flag_own_car'] == 'Y'
df1['flag_own_realty'] = df0['flag_own_realty'] == 'Y'
df1['flag_own_car'][:5]


# In[26]:


cat_vars_0  = df1.dtypes[ df1.dtypes == 'object' ]
cat_vars_0


# In[28]:


import df_one_hot_encode
oh_enc = df_one_hot_encode.DfOneHotEncoder( cat_vars_0.index )

oh_enc.fit( df1 )
df1 = oh_enc.transform( df1, drop_old=True )


# In[29]:


# Convert flag columns to bool
for col in df1.columns :
    if col.startswith( 'flag_' ) or col.startswith( 'reg_') : 
        df1[col] = (df1[col] == 1)
        print( df1[col].value_counts() )


# In[ ]:





# In[ ]:


# For float64 columns impute  NaNs with median 
for col in df1.select_dtypes('float64').columns : 
    if df1[col].isnull().sum() > 0 :  
        median = df1[col].median() 
        df1[col] = df1[col].fillna(  median )
# check how well XGBoost handles NaNs


# In[32]:


from sklearn.model_selection import train_test_split
train, test = train_test_split( df1, train_size = 0.8, test_size = 0.2 )

y_train = train['target']
y_test  = test['target']

x_train = train.loc[ : , train.columns != 'target']
x_test  = test .loc[ : , test.columns != 'target']
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)    # (246008, 181) (61503, 181) (246008,) (61503,)


# In[39]:


y_test.unique()


# In[37]:


ratio = (y_train == 0).sum()/ (y_train == 1).sum()
print("ratio of 0 to 1 in target column ", ratio)
print("Postive examples in train set: {}".format(np.sum(y_train==0)))
print("Negative examples in train set: {}".format(np.sum(y_train==1)))

print("Postive examples in test set: {}".format(np.sum(y_test==0)))
print("Negative examples in test set: {}".format(np.sum(y_test==1)))


# In[ ]:


import xgboost as xgbd

#Loading data into DMatrices
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


# In[ ]:





# In[38]:


# build model XGBoost with some specific hyper parameters
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


clf = XGBClassifier(n_estimators=1000, objective='binary:logistic', gamma=0.1, subsample=0.5, scale_pos_weight=ratio )
clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=10)


# In[40]:


# previosly best iteration at AUC=0.68. Cross validation again on the entire dataset with the best tree from prev training
n_estimators = clf.best_ntree_limit
clf = XGBClassifier(n_estimators=n_estimators, objective='binary:logistic', gamma=0.1, subsample=0.5, scale_pos_weight=ratio )

# fit on entire train data
clf.fit(x_train, y_train, eval_set=[(x_train, y_train)], eval_metric='auc')


# In[45]:


# get AUC score for both train and test data sets
def roc_auc_scorer( clf, x, y ) :
    y_pred = clf.predict_proba(x)[:,1]
    return roc_auc_score( y, y_pred )

roc_auc_scorer( clf, x_train, y_train ), roc_auc_scorer( clf, x_test, y_test )


# In[47]:


clf.get_xgb_params()


# In[48]:


# plot XGBoost feature importance
from xgboost import plot_importance
from matplotlib import pyplot

plot_importance(clf)
pyplot.show()


# In[56]:


import matplotlib.pylab as plt
feat_imp = pd.Series(clf.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')


# In[ ]:




