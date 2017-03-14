# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:54:03 2017

@author: glen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Titanic as ti
from sklearn.cross_validation import train_test_split
file = 'input/train.csv'
cleanData = ti.featureEngineering(file)
train, test = train_test_split(cleanData, test_size=.2, random_state=100)
col = ['XGBoost',
       'RandomForest',
       'ExtraTrees',
       'AdaBoost',
       'GradientBoost',
       'SupportVector',
       'Survived']
# Create empty dataframes
testingTruth = pd.DataFrame({'Run': [],
                             'Model': [],
                             'Accuracy': []},
                             columns=['Run',
                                      'Model',
                                      'Accuracy'])
estimators = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
learnrate = [[.7,.1], [.6,.1], [.5,.1], [.4,.1],
             [.3,.1], [.2,.1], [.1,.1], [1,.1],
             [.09,.1], [.08,.1], [.07,.1] ]
for run in range(0, 1):
    
    print ('Test run {}'.format(run))
    train, test = train_test_split(cleanData, test_size=.2)
    training = ti.TrainingEngine(train, seed=100)    
    training.params['RFEstimators'] = 90
    training.params['ETEstimators'] = 30
    training.params['ABEstimators'] = 120
    training.params['GBEstimators'] = 20
    training.params['XGEstimators'] = 160
    training.defineModels()
    
    print(' Train train and predict')
    indata = training.firstLevelTrainer()
    secondModel = training.secondLevelTrainer(indata)
    print(' Accuracy for models used to train data')
    print(training.accuracy)
    
    print(' Test predictions')
    indata = training.firstLevelPredict(indata=test.copy(), models=training.models)
    indata = training.secondLevelPredict(indata, secondModel)
    del training
    print(' Build accuracy table for testing data')
    df = ti.BuildTruthTable(indata[col], run, 'Survived')
    testingTruth = testingTruth.append(df)
    #df.to_csv('testtruth.csv', header=False, mode='a')
    del df