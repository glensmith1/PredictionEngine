# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:54:03 2017

@author: glen
"""
import pandas as pd
import pprint as pp
import Titanic as ti
import xgboost as xgb
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import precision_recall_fscore_support

def errorRate(model, actual, indata):
    total = len(indata)
    error = len(indata[indata[model] != indata[actual]])
    return error/total

file = 'input/train.csv'
print('Feature Engineerng')
inputData = ti.featureEngineering(file)
train, test = train_test_split(inputData, test_size=.2, random_state=100)

#train, test = train_test_split(cleanData, test_size=.2, random_state=100)
col = ['XGBoost',
       'RandomForest',
       'ExtraTrees',
       'AdaBoost',
       'GradientBoost',
       'SupportVector',
       'Survived']
accuracyForRun = {}

print('Training Run')
training = ti.TrainingEngine(train, seed=100)
actual = train['Survived']
indata = training.firstLevelTrainer()
indata = training.secondLevelTrainer(indata)
accuracyForRun['Train']  = {model: errorRate('Survived', model, indata)
                            for model in col if model != 'Survived'}


print('Testing Run')
actual = test['Survived']
final = training.firstLevelPredict(test.copy(), training.models)
final = training.secondLevelPredict(final, training.models['XGBoost'])
accuracyForRun['Test'] = {model: errorRate('Survived', model, final)
                          for model in col if model != 'Survived'}
errRate = pd.DataFrame(accuracyForRun, columns=['Train', 'Test'])