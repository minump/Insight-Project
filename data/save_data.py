# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:20:30 2020

@author: minum
"""
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class SaveData:
    
    def __init__(self, data):
        self.df=data
        
    def save_train_test_data(self):
        # split into train val and test
        train, test = train_test_split( self.df, test_size = 0.2, random_state=42 )
        #print(train.shape, test.shape) #(39720, 175) (9930, 175)
        train = shuffle(train)
        
        partial_train, val = train_test_split(train, test_size=0.2, random_state=42)
        print("train, val and test shape")
        print(partial_train.shape, val.shape, test.shape) # (31776, 175) (7944, 175) (9930, 175)

        y_train = partial_train['target']
        y_val = val['target']
        y_test  = test['target']

        x_train = partial_train.loc[ : , partial_train.columns != 'target']
        x_val =  val.loc[:,val.columns!='target']
        x_test  = test.loc[ : , test.columns != 'target']
        print("x_train, x_val, x_test, y_train, y_val, y_test shape")
        print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)    
        #(31776, 174) (7944, 174) (9930, 174) (31776,) (7944,) (9930,)


        x_train.to_hdf('data/x_train.h5', key='df', mode='w')
        y_train.to_hdf('data/y_train.h5', key='df', mode='w')
        x_val.to_hdf('data/x_val.h5', key='df', mode='w')
        y_val.to_hdf('data/y_val.h5', key='df', mode='w')
        x_test.to_hdf('data/x_test.h5', key='df', mode='w')
        y_test.to_hdf('data/y_test.h5', key='df', mode='w')