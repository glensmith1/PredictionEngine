#! Titanic.py
# Load in our libraries
import pandas as pd
import numpy as np
import re
import xgboost as xgb
import warnings as warn
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV

def featureEngineering(file, ageCut=0, nlCut=0, notUsed=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']):
    ''' Loads a file for data enigineering purposes '''

    def catNameLength(data, nlCut):
        ''' Create name_length and put into categories '''
        nameLength = [len(name) for name in data]
        if nlCut < 1:
            nlCut = max(nameLength)
        return pd.cut(nameLength, nlCut, labels=np.array(range(nlCut))).astype(int)

    def catFare(indata):
        ''' Categorize fare (some are null and are set to defaults) '''
        data = indata.copy()
        nans = np.isnan(data)
        medn = data.median()
        data[nans] = medn
        data = [3 if fare > 31 else 2 if fare < 14.54 else 1 if fare > 7.91 else 0
                for fare in data]
        return data
        
    def catAge(indata, ageCut):
        ''' Categorize ages (some are null and are set to defaults)  '''
        data   = indata.copy()
        ageAvg = data.mean()
        ageStd = data.std()
        nans   = np.isnan(data)
        toRepl = nans.sum()
        data[nans] = np.random.randint(ageAvg-ageStd, ageAvg+ageStd, toRepl)
        if ageCut < 1:
            ageCut = data.max().astype(int)
        return pd.cut(data, ageCut, labels=np.array(range(ageCut))).astype(int)
        
    def catTitle(data):
        ''' Create a title and categorize '''
        titleMap = {"Unk":0,
                    "Mr":1,
                    "Miss":2,
                    "Mlle":2,
                    "Ms": 2,
                    "Mme":3,
                    "Mrs":3,
                    "Master":4,
                    "Lady":5,
                    "Countess":5,
                    "Capt":5,
                    "Col":5,
                    "Don":5,
                    "Dr":5,
                    "Major":5,
                    "Rev":5,
                    "Sir":5,
                    "Jonkheer":5,
                    "Dona":5}
        titleLst = np.array([re.search(' ([A-Za-z]+)\.', name).group(1)
                             for name in data])
        titleCat = np.array([titleMap[title] for title in titleLst])
        titleNan = np.isnan(titleCat)
        titleCat[titleNan] = 0
        return titleCat
 
    if isinstance(file, pd.DataFrame):
        data = file.copy()
    else:
        try:
            data = pd.read_csv(file).copy()
        except:
            warn.warn('file must be a dataframe or a file')
            return
        
    data['Name_length'] = catNameLength(data['Name'], nlCut)
    data['Has_Cabin'] = data['Cabin'].map(lambda x: 0 if type(x) == float else 1)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = data['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, np.nan: 0} )
    data['Fare'] = catFare(data['Fare'])
    data['Age'] = catAge(data['Age'], ageCut)
    data['Title'] = catTitle(data['Name'])
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} )
    data = data.drop(notUsed, axis = 1)
    
    return data
    
class TrainingEngine(object):
    ''' Trains the data '''
   
    def __init__(self, data, folds=5, seed=0, predCol='Survived'):
        
        self.data       = data.copy()
        self.nTrain     = len(self.data)
        self.predCol    = predCol
        self.featureIdx = [column for column in data.columns if column != predCol] 
        self.kf         = KFold(self.nTrain, n_folds=folds, random_state=seed)
        self.params     = {'Seed': seed}
        self.tuneXGB    = False
        self.levelOneEstimators = ['RandomForest',
                                   'ExtraTrees',
                                   'AdaBoost',
                                   'GradientBoost',
                                   'SupportVector']
        self.levelTwoEstimators = ['XGBoost']
        self.earlyStopping = 50
        self.trainingAction = ['Tuning', 'Tuning']

        # Classifiers
        self.models = {'RandomForest':  RandomForestClassifier(n_jobs=-1,
                                                               random_state=seed),
                       'ExtraTrees':    ExtraTreesClassifier(n_jobs=-1,
                                                             random_state=seed),
                       'AdaBoost':      AdaBoostClassifier(random_state=seed),
                       'GradientBoost': GradientBoostingClassifier(random_state=seed),
                       'SupportVector': SVC(kernel='linear',
                                            random_state=seed),
                       'XGBoost':       XGBClassifier(nthread=-1,
                                                      seed=seed)}
                            
        # Parameters
        self.params = {'RandomForest':  {'max_depth': [3,6,9],
                                         'n_estimators': [80,75,70]},
                       'ExtraTrees':    {'max_depth': [7,8,9],
                                         'n_estimators': [15,30,45]},
                       'AdaBoost':      {'learning_rate': [1,.7,.4],
                                         'n_estimators': [50,100,150]},
                       'GradientBoost': {'learning_rate': [.1,.4,.7],
                                         'max_depth': [3,6,9],
                                         'n_estimators': [100,200,300]},
                       'SupportVector': {'C': [.5,.75,.1,.05,.025]},
                       'XGBoost':       {'n_estimators':[50,100,150],
                                         'min_child_weight': [1,2,3], 
                                         'max_depth': [2,5,8],
                                         'gamma': [0,.4,.8],
                                         'learning_rate':[.3,.4,.5]}}
                       
        # Hold output of grid search
        self.grid = {}
 

    def tune(self, model, x, y, scoring='precision'):
        ''' tune models '''
        print(model, end=' ')
        estimator = self.models[model]
        tuning = self.params[model]
        clf = GridSearchCV(estimator, tuning, cv=len(self.kf), scoring=scoring)
        clf.fit(x, y)
        return clf
    
    def firstLevelTrainer(self):
        ''' This trains the data for the first level models '''
        
        def oneModel(model):
            clf = self.models[model]
            if self.trainingAction[0] == 'Tuning':
                tclf = self.tune(model, x_train, y_train)
                clf.set_params(**tclf.best_params_)
                self.grid[model] = tclf 
            [clf.fit(x_train[index[1]], y_train[index[1]]) for index in self.kf]
            
        # Some parameters for training
        indata   = self.data.copy()
        x_train  = indata[self.featureIdx].values
        y_train  = indata[self.predCol].values
        
        # Train all first level model
        print('{} level 0:'.format(self.trainingAction[0]), end=' ')
        [oneModel(model) for model in self.levelOneEstimators]
        print()
        
        return self.firstLevelPredict(indata=indata)

    def firstLevelPredict(self, indata, models=None):
        ''' Prediction using first level models '''
        
        def oneModel (model):
            clf = models[model]
            predictions = np.array([clf.predict(features) 
                                    for _, test_index in self.kf])
            predictions = predictions.mean(axis=0).astype(int)
            return predictions.flatten()
            
        # Some parameters for prediction
        if not models:
            models = self.models
        features = indata[self.featureIdx]
  
        # Put changes into pandas dataframe
        predictions = {model: oneModel(model) for model in self.levelOneEstimators}
        {indata.insert(0, column=model, value=series.ravel())
         for model, series in predictions.items()}

        return indata

    def secondLevelTrainer(self, indata, stackModel='XGBoost'):
        ''' Train second level using second level model '''

        # Some parameters for training
        gbm     = self.models[stackModel]
        x_train = indata[self.levelOneEstimators].values
        y_train = indata[self.predCol].values
        
        # Tune
        if self.trainingAction[1] == 'Tuning':
            print('Tuning level 1:', end=' ')
            tclf = self.tune(stackModel, x_train, y_train)
            gbm.set_params(**tclf.best_params_)
            self.grid[stackModel] = tclf 

        # Train
        gbm = gbm.fit(x_train, y_train)
        self.models[stackModel] = gbm
        
        # Predict for this model
        indata = self.secondLevelPredict(indata, gbm)

        return indata

    def secondLevelPredict(self, indata, gbm, stackModel='XGBoost'):
        ''' Predictions using second level model '''

        # Some parameters for prediction        
        features    = indata[self.levelOneEstimators].values

        # Make predictions
        predictions = gbm.predict(features)

        # Insert predictions into predictions tables
        indata = indata.drop(stackModel, axis=1, errors='ignore')
        indata.insert(0, stackModel, predictions)

        return indata