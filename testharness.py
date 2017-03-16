# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:54:03 2017

@author: glen
"""
import pandas as pd
import Titanic as ti
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

file = 'input/train.csv'
inputTrain = pd.read_csv(file)
trainSample, testSample = train_test_split(inputTrain, test_size=.2, random_state=100)

#train, test = train_test_split(cleanData, test_size=.2, random_state=100)
col = ['XGBoost',
       'RandomForest',
       'ExtraTrees',
       'AdaBoost',
       'GradientBoost',
       'SupportVector',
       'Survived']
cutlist = [1,2,3,4,5,6,7,8,9,0]
randomForestEstimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
estimators = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
learnrate = [[.7,.1], [.6,.1], [.5,.1], [.4,.1],
             [.3,.1], [.2,.1], [.1,.1], [1,.1],
             [.09,.1], [.08,.1], [.07,.1] ]
runs = 10
trainingAccuracy = []
testingAccuracy = []
finalModels = []
train = ti.featureEngineering(trainSample, ageCut=9, nlCut=9)
for run in range(0, 20):

    print ('Train run {}'.format(run))
    training = ti.TrainingEngine(train, seed=100)
    training.params['XGEstimators'] = 23
    training.params['RFEstimators'] = 50
    training.defineModels()
    actual = train['Survived']
    training.tuneXGB = False
    indata = training.firstLevelTrainer()
    indata = training.secondLevelTrainer(indata)
    accuracyForRun = {model: accuracy_score(actual, indata[model]) 
                      for model in col if model != 'Survived'}
    accuracyForRun['XGBoost'] = accuracy_score(actual, indata['XGBoost'])
    trainingAccuracy.append(accuracyForRun)
    finalModels.append(training.stackModel)

#pd.DataFrame(trainingAccuracy).max()

test = ti.featureEngineering(testSample, ageCut=9, nlCut=9)
for run in range(0, 20):
    
    print('Test run {}'.format(run))
    actual = test['Survived']
    final = training.firstLevelPredict(indata=test.copy(), models=training.models)
    gbmModel = finalModels[run]
    final = training.secondLevelPredict(indata=final, gbm=gbmModel)
    accuracyForRun = {model: accuracy_score(actual, final[model])
                      for model in col if model != 'Survived'}
    accuracyForRun['XGBoost'] = accuracy_score(actual, indata['XGBoost'])
    testingAccuracy.append(accuracyForRun)
