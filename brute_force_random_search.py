from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from itertools import product, combinations
import math
import numpy as np
from scipy import stats
import copy
import random
import operator
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
            estimator.set_params(**params)
            self.ensemble.append(Estimator(classifier=estimator, random_state=self.random_state, fitness=best_fitness_classifiers[classifiers_fitness_it]))
            classifiers_fitness_it = classifiers_fitness_it + 1
        for classifier in self.ensemble:
            classifier.fit(X, y)
        
    def fit(self, X, y, all_possible_ensembles):
        len_all_possible_ensembles = len(all_possible_ensembles)
        len_y = len(y)
        result_dict = dict()
        selected_ensemble = np.zeros([len_all_possible_ensembles],dtype=bool)
        random.seed(self.random_state)
        kf = KFold(n_splits=5, random_state=self.random_state)
        best_ensemble_fitness = np.zeros([len_y])
        best_fitness_classifiers = np.zeros([self.n_estimators])
        i = 0
        while i < (len_all_possible_ensembles):
            # a matrix with all observations vs the prediction of each classifier
            classifiers_predictions = np.zeros([len_y,self.n_estimators])
            # sum the number of right predictions for each classifier
            classifiers_right_predictions = np.zeros([self.n_estimators])
            ensemble_fitness = np.zeros([len_y])
            classifier_id = 0
            if i <= self.stop_time:
                classifiers = random.choice(range(len_all_possible_ensembles))
                if selected_ensemble[classifiers] == False:
                    selected_ensemble[classifiers] = True
                    for cl in range(0, self.n_estimators):
                        classifier = all_possible_ensembles[classifiers][0][cl][0]
                        params = all_possible_ensembles[classifiers][0][cl][1]
                        classifier.set_params(**params)
                        y_pred = np.zeros([len_y])
                        #k-fold cross-validation
                        for train, val in kf.split(X):
                            classifier.fit(X[train], y[train])
                            y_pred[val] = classifier.predict(X[val])
                            for idx_obj in val: 
                                classifiers_predictions[idx_obj][classifier_id] = y_pred[idx_obj]
                        classifiers_right_predictions[classifier_id] = np.equal(y_pred, y).sum()
                        classifier_id = classifier_id + 1
                    #the ensemble make the final prediction by majority vote for accuracy
                    majority_voting = stats.mode(classifiers_predictions, axis=1)[0]
                    majority_voting = [int(j[0]) for j in majority_voting]
                    ensemble_fitness = np.equal(majority_voting,y)
                    #select the most accurate ensemble
                    if(ensemble_fitness.sum() > best_ensemble_fitness.sum()):
                        best_ensemble_fitness = ensemble_fitness
                        best_fitness_classifiers = classifiers_right_predictions
                        ensemble = all_possible_ensembles[classifiers]
                    end_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
                    result_dict.update({i: {"end_time":end_time,"best_fitness_ensemble":best_ensemble_fitness.sum(), "ensemble":ensemble, "best_fitness_classifiers":best_fitness_classifiers}})
                    i = i +1
            else:
                return result_dict
    
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
