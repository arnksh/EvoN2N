# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:51:09 2021

@author: ARUN
"""


from __future__ import print_function

from sklearn import svm

import numpy as np

import scipy.io as sio

from sklearn import metrics
import xlsxwriter

#from sklearn.model_selection import train_test_split
dataFolder = 'GFD'
#tarData = ['tarData_1','tarData_2','tarData_3','tarData_4'];

workbook = xlsxwriter.Workbook(f'./logs_GFD/EvoDCNN_{dataFolder}.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(1,2, f'{dataFolder}')

c = 0
for i in range(30,90,20):
    f0=sio.loadmat(f'../{dataFolder}/h_b_30hz_{i}.mat')
    A = f0['Y'][0,0]
    x_train = A['training_inputs']
    y_train = A['training_results'][:,0]
    x_test = A['test_inputs']
    y_test = A['test_results'][:,0]
    
    if np.min(y_train)==1:
        y_train = y_train-1
        y_test = y_test-1
        
    clf = svm.SVC(kernel='linear')
    
    clf.fit(x_train, y_train)

#Predict the response for test dataset
    y_pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred);
    worksheet.write(c+2,2, f'h_b_30hz_{i}')
    worksheet.write(c+2,4, accuracy*100)
    c+=1
        
workbook.close()


