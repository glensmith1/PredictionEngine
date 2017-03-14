#! Titanic.py
# Load in our libraries
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

def featureEngineering(file, ageCut=10, nlCut=10, notUsed=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']):
    ''' Loads a file for data enigineering purposes '''

    def catNameLength(data):
        ''' Create name_length and put into categories '''
        nameLength = [len(name) for name in data]
        return pd.cut(nameLength, nlCut, labels=np.array(range(nlCut)))

    def catFare(indata):
        ''' Categorize fare (some are null and are set to defaults) '''
        data = indata.copy()
        nans = np.isnan(data)
        medn = data.median()
        data[nans] = medn
        data = [3 if fare > 31 else 2 if fare < 14.54 else 1 if fare > 7.91 else 0
                for fare in data]
        return data
        
    def catAge(indata):
        ''' Categorize ages (some are null and are set to defaults)  '''
        data   = indata.copy()
        ageAvg = data.mean()
        ageStd = data.std()
        nans   = np.isnan(data)
        toRepl = nans.sum()
        data[nans] = np.random.randint(ageAvg-ageStd, ageAvg+ageStd, toRepl)
        return pd.cut(data, ageCut, labels=np.array(range(ageCut)))
        
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
 
    data = pd.read_csv(file).copy()
    data['Name_length'] = catNameLength(data['Name'])
    data['Has_Cabin'] = data['Cabin'].map(lambda x: 0 if type(x) == float else 1)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = data['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, np.nan: 0} )
    data['Fare'] = catFare(data['Fare'])
    data['Age'] = catAge(data['Age'])
    data['Title'] = catTitle(data['Name'])
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} )
    data = data.drop(notUsed, axis = 1)
    
    return data
    
class SklearnHelper(object):
    ''' Extends the sklearn classifiers '''
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_

class TrainingEngine(object):
    ''' Trains the data '''
    def __init__(self, data, folds=5, seed=0, predCol='Survived'):
        self.data       = data.copy()
        self.nTrain     = len(self.data)
        self.predCol    = predCol
        self.featureIdx = [column for column in data.columns if column != predCol] 
        self.kf         = KFold(self.nTrain, n_folds=folds, random_state=seed)
        self.params     = {'Seed': seed,
                           'RFEstimators': 90,
                           'ETEstimators': 30,
                           'ABEstimators': 120,
                           'GBEstimators': 40,
                           'XGEstimators': 160,
                           'RFMaxDepth': 6,
                           'ETMaxDepth': 8,
                           'GBMaxDepth': 5,
                           'XGMaxDepth': 4,
                           'LearnRate': [0.7, 0.1],
                           'MinLeaf': 2,
                           'Verbose': 0,
                           'MinChildWeight': 2,
                           'Gamma': 0.9}
        self.defineModels()

    def defineModels(self):
        ''' This method sets the models this will use '''
        # First Level
        self.models = {'RandomForest': SklearnHelper(clf=RandomForestClassifier,
                                                     seed=self.params['Seed'],
                                                     params={'n_jobs': -1,
                                                             'n_estimators': self.params['RFEstimators'],
                                                             'max_depth': self.params['RFMaxDepth'],
                                                             'min_samples_leaf': self.params['MinLeaf'],
                                                             'verbose': self.params['Verbose']}),
                         'ExtraTrees': SklearnHelper(clf=ExtraTreesClassifier,
                                                     seed=self.params['Seed'],
                                                     params={'n_jobs': -1,
                                                             'n_estimators': self.params['ETEstimators'],
                                                             'max_depth': self.params['ETMaxDepth'],
                                                             'min_samples_leaf': self.params['MinLeaf'],
                                                             'verbose': self.params['Verbose']}),
                           'AdaBoost': SklearnHelper(clf=AdaBoostClassifier,
                                                     seed=self.params['Seed'],
                                                     params={'n_estimators': self.params['ABEstimators'],
                                                             'learning_rate': self.params['LearnRate'][0]}),
                      'GradientBoost': SklearnHelper(clf=GradientBoostingClassifier,
                                                     seed=self.params['Seed'],
                                                     params={'n_estimators': self.params['GBEstimators'],
                                                             'max_depth': self.params['GBMaxDepth'],
                                                             'min_samples_leaf': self.params['MinLeaf'],
                                                             'verbose': self.params['Verbose']}),                     
                      'SupportVector': SklearnHelper(clf=SVC,
                                                     seed=self.params['Seed'],
                                                     params={'kernel' : 'linear',
                                                             'C' : 0.025}),
                            'XGBoost': SklearnHelper(clf=XGBClassifier,
                                                     seed=self.params['Seed'],
                                                     params={'nthread': -1,
                                                             'n_estimators': self.params['XGEstimators'],
                                                             'max_depth': self.params['XGMaxDepth'],
                                                             'learning_rate': self.params['LearnRate'][1],
                                                             'min_child_weight': self.params['MinChildWeight'],
                                                             'gamma': self.params['Gamma'],                        
                                                             'subsample': 0.8,
                                                             'colsample_bytree': 0.8,
                                                             'objective': 'binary:logistic',
                                                             'scale_pos_weight': 1})}



    def firstLevelTrainer(self, exclude=None):
        ''' This trains the data for the first level models '''
        def oneModel(clf):
            [clf.train(x_train[index], y_train[index]) for index, _ in self.kf]

        # Some parameters for training
        indata   = self.data.copy()
        models   = self.models
        x_train  = indata[self.featureIdx].values
        y_train  = indata[self.predCol].values
        
        # Train all first level model
        {model: oneModel(clf) for model, clf in models.items()}
        
        # Training prediction to get accuracy score
        predictions = self.firstLevelPredict()
        self.accuracyScore = {model: accuracy_score(y_train, predictions[model])
                             for model in predictions.columns}
            
        return predictions

    def firstLevelPredict(self, indata=None, models=None):
        ''' Prediction using first level models '''
        def oneModel (clf):
            predictions = np.array([clf.predict(features) 
                                    for _, test_index in self.kf])
            predictions = predictions.mean(axis=0).astype(int)
            return predictions.flatten()
            
        # Some parameters for prediction
        if not isinstance(indata, pd.DataFrame):
            indata = self.data.copy()
        if not models:
            models = self.models
        features = indata[self.featureIdx]
  
        # Predict for each model
        predictions = {model: oneModel(clf) for model, clf in models.items()}

        # Put changes into pandas dataframe
        {indata.insert(0, column=model, value=series.ravel())
         for model, series in predictions.items()}

        return indata

    def secondLevelTrainer(self, indata, stackModel='XGBoost'):
        ''' Train second level using second level model '''

        # Some parameters for training
        gbm     = self.model[stackModel]
        columns = [model for model in self.models if model not in stackModel]
        x_train = np.stack(indata[columns].to_dict('list').values(), axis=1)
        y_train = indata[self.predCol].values

        # Train this model
        gbm = gbm.fit(x_train, y_train)
        
        # Predict for this model
        indata = self.secondLevelPredict(indata, stackModel)
        self.accuracyScore[stackModel] = accuracy_score(y_train, indata[stackModel])
        
        # Save this model
        self.stackModel = gbm

        return indata

    def secondLevelPredict(self, indata, model, stackModel='XGBoost'):
        ''' Predictions using second level model '''

        # Some parameters for prediction        
        columns     = [model for model in self.models if model not in stackModel]
        features    = np.stack(indata[columns].to_dict('list').values(), axis=1)
        
        # Make predictions
        predictions = gbm.predict(features)

        # Insert predictions into predictions tables
        indata = indata.drop(stackModel, axis=1, errors='ignore')
        indata.insert(0, self.stackModel, predictions)

        return indata
        
def BuildTruthTable(indata, inrun, actual):
    ''' Function to build a truth table for this input data '''

    def getSum(model):
        return accuracy_score(indata[actual], indata[model])

    models = [model for model in indata.columns if model not in actual]
        
    return {inrun, {model: getSum(model) for model in models}}