# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:54:03 2017

@author: glen
"""

import pprint
import Titanic as ti
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
file = 'input/train.csv'
cleanData = ti.featureEngineering(file, ageCut=20, nlCut=20)
train, test = train_test_split(cleanData, test_size=.2, random_state=100)
y_train = train['Survived']
y_test  = test['Survived']
col = ['XGBoost',
       'RandomForest',
       'ExtraTrees',
       'AdaBoost',
       'GradientBoost',
       'SupportVector',
       'Survived']
estimators = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
learnrate = [[.7,.1], [.6,.1], [.5,.1], [.4,.1],
             [.3,.1], [.2,.1], [.1,.1], [1,.1],
             [.09,.1], [.08,.1], [.07,.1] ]
for run in range(0, 1):
    
    print ('Test run {}'.format(run))
    training = ti.TrainingEngine(train, seed=100)    
    training.params['RFEstimators'] = 90
    training.params['ETEstimators'] = 30
    training.params['ABEstimators'] = 120
    training.params['GBEstimators'] = 20
    training.params['XGEstimators'] = 160
    training.defineModels()
    print(' Training parameters used:')
    pprint.pprint(training.params)
    
    indata = training.firstLevelTrainer()
    levelOneAccuracy = {model: accuracy_score(y_train, indata[model]) 
                        for model in col if model != 'Survived'}
    training.stackModel = ti.XGBClassifier()
    indata = training.secondLevelTrainer(indata)
    levelTwoAccuracy = {'XGBoost': accuracy_score(y_train, indata['XGBoost'])}
    print(' Accuracy in training:')
    pprint.pprint(levelOneAccuracy, indent=4)
    pprint.pprint(levelTwoAccuracy, indent=4)
    
    final = training.firstLevelPredict(indata=test.copy(), models=training.models)
    levelOneAccuracy = {model: accuracy_score(y_test, final[model]) 
                        for model in col if model != 'Survived'}
    final = training.secondLevelPredict(indata=final, gbm=training.stackModel)
    levelTwoAccuracy = {'XGBoost': accuracy_score(y_test, final['XGBoost'])}
    del training
    print(' Accuracy in testing:')
    pprint.pprint(levelOneAccuracy, indent=4)
    pprint.pprint(levelTwoAccuracy, indent=4)