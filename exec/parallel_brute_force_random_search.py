from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from itertools import product, combinations
import math
import numpy as np
import pandas as pd
from scipy import stats
import copy
import random
import operator
import time
import sys, getopt
import math
import multiprocessing
from joblib import Parallel, delayed, load, dump
import tempfile
import os
import shutil

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from joblib import Memory
memory = Memory("./tmpmemoryjoblib", verbose=0)

@memory.cache
def train_clf(classifier, params, X, y, random_state):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    clf = getattr(sys.modules[__name__], classifier)()
    clf.set_params(**params)
    y_pred = np.zeros([len(y)])
    #k-fold cross-validation
    kf = KFold(n_splits=5, random_state=random_state)
    for train, val in kf.split(X):
        clf.fit(X[train], y[train])
        y_pred[val] = clf.predict(X[val])
    return y_pred

class Estimator:
    def __init__(self, classifier=None, random_state=None, fitness=0):
        self.classifier = classifier
        self.random_state = random_state
        self.fitness = fitness

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)


class BruteForceEnsembleClassifier:
    def __init__(self, stop_time = 100, n_estimators = 10, random_state = None):
        self.n_estimators = n_estimators
        self.ensemble = []
        self.stop_time = stop_time
        self.random_state = random_state
        random.seed(self.random_state)

    def estimators_pool(self, estimator_grid):
        for estimator, param_grid in estimator_grid.items():
            items = sorted(param_grid.items())
            if not items:
                yield (estimator, {})
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield (estimator, params)
                    
    def fit_ensemble(self, X, y, ensemble, best_fitness_classifiers):
        classifiers_fitness_it = 0
        for estimator, params in ensemble:
            estimator = getattr(sys.modules[__name__], estimator)()
            estimator.set_params(**params)
            self.ensemble.append(Estimator(classifier=estimator, 
                                           random_state=self.random_state, 
                                           fitness=best_fitness_classifiers[classifiers_fitness_it]))
            classifiers_fitness_it = classifiers_fitness_it + 1
        for classifier in self.ensemble:
            classifier.fit(X, y)
        
    #def parallel_fit(self, X, y, all_possible_ensembles, classifiers):
    def parallel_fit(self, X, y, classifiers):
        now = time.time()
        struct_now = time.localtime(now)
        mlsec = repr(now).split('.')[1][:3]
        start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
        time_aux = int(round(now * 1000))
        len_y = len(y)
        result_dict = dict()
        random.seed(self.random_state)
        # a matrix with all observations vs the prediction of each classifier
        classifiers_predictions = np.zeros([self.n_estimators, len_y])
        # sum the number of right predictions for each classifier
        classifiers_right_predictions = np.zeros([self.n_estimators])
        ensemble_fitness = np.zeros([len_y])
        classifier_id = 0
        for cl in range(0, self.n_estimators):
            #classifier = all_possible_ensembles[classifiers][0][cl][0]
            classifier = classifiers[0][cl][0]
            #params = all_possible_ensembles[classifiers][0][cl][1]
            params = classifiers[0][cl][1]
            y_pred = train_clf(classifier, params, X, y, self.random_state)
            classifiers_predictions[classifier_id][:] = y_pred
            classifiers_right_predictions[classifier_id] = np.equal(y_pred, y).sum()
            classifier_id = classifier_id + 1
        #the ensemble make the final prediction by majority vote for accuracy
        majority_voting = stats.mode(classifiers_predictions, axis=0)[0]
        majority_voting = [int(j[0]) for j in majority_voting]
        ensemble_fitness = np.equal(majority_voting,y)
        ensemble = classifiers
        now = time.time()
        struct_now = time.localtime(now)
        mlsec = repr(now).split('.')[1][:3]
        end_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
        total_time = (int(round(now * 1000)) - time_aux)
        result_dict.update({"start_time":start_time,
                            "end_time":end_time,
                            "total_time_ms":total_time,
                            "fitness_ensemble":ensemble_fitness.sum(),
                            "ensemble":ensemble,
                            "fitness_classifiers":classifiers_right_predictions})
        return result_dict
                
    def fit(self, X, y, all_possible_ensembles, selected_ensemble, n_cores):
        backend = 'loky'
        #result = Parallel(n_jobs=n_cores, backend=backend)(delayed(self.parallel_fit)(X, y, all_possible_ensembles, item) for index, item in zip(range(0, self.stop_time), selected_ensemble))
        result = Parallel(n_jobs=n_cores, backend=backend)(delayed(self.parallel_fit)(X, y, all_possible_ensembles[item]) for index, item in zip(range(0, self.stop_time), selected_ensemble))
        return result
    
    def predict(self, X):
        len_X = len(X)
        predictions = np.zeros((self.n_estimators, len_X))
        y = np.zeros(len_X)
        for estimator in range(0, self.n_estimators):
            predictions[estimator] = self.ensemble[estimator].predict(X)
        for i in range(0, len_X):
            pred = {}
            for j in range(0, self.n_estimators):
                if predictions[j][i] in pred:
                    pred[predictions[j][i]] += self.ensemble[j].fitness
                else:
                    pred[predictions[j][i]]  = self.ensemble[j].fitness
            y[i] = max(pred.items(), key=operator.itemgetter(1))[0]
        return y