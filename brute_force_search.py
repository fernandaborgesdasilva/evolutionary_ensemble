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
    def __init__(self, algorithms, stop_time = 100, n_estimators = 10, random_state = None):
        self.n_estimators = n_estimators
        self.ensemble = []
        self.stop_time = stop_time
        self.algorithms = algorithms
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
        
    def fit(self, X, y):
        result_dict = dict()
        random.seed(self.random_state)
        kf = KFold(n_splits=5, random_state=self.random_state)
        best_ensemble_fitness = np.zeros([len(y)])
        best_fitness_classifiers = np.zeros([self.n_estimators])
        for i, classifiers in enumerate(combinations(self.estimators_pool(self.algorithms),self.n_estimators)):
            # a matrix with all observations vs the prediction of each classifier
            classifiers_predictions = np.zeros([len(y),self.n_estimators])
            # sum the number of right predictions for each classifier
            classifiers_right_predictions = np.zeros([self.n_estimators])
            ensemble_fitness = np.zeros([len(y)])
            classifier_id = 0
            if i >= self.stop_time:
                break
            for classifier, params in classifiers:
                classifier.set_params(**params)
                count_right_answers = 0
                #k-fold cross-validation
                for train, val in kf.split(X):
                    classifier.fit(X[train], y[train])
                    y_pred = classifier.predict(X[val])
                    count_right_answers = count_right_answers + np.equal(y_pred, y[val]).sum()
                    for observation, result in enumerate(y_pred):
                        classifiers_predictions[observation][classifier_id] = result
                classifiers_right_predictions[classifier_id] = count_right_answers
                classifier_id = classifier_id + 1
            #the ensemble make the final prediction by majority vote for accuracy
            majority_voting = stats.mode(classifiers_predictions, axis=1)[0]
            majority_voting = [int(j[0]) for j in majority_voting]
            for iterator in range(0, len(y)):
                ensemble_fitness[iterator] = np.equal(majority_voting[iterator],y[iterator])
            #select the most accurate ensemble
            if(ensemble_fitness.sum() > best_ensemble_fitness.sum()):
                best_ensemble_fitness = ensemble_fitness
                best_fitness_classifiers = classifiers_right_predictions
                ensemble = classifiers
            end_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            result_dict.update({i: {"end_time":end_time,"best_fitness_ensemble":best_ensemble_fitness.sum(), "ensemble":ensemble, "best_fitness_classifiers":best_fitness_classifiers}})
        return result_dict
    
    def predict(self, X):
        predictions = np.zeros((self.n_estimators, len(X)))
        y = np.zeros(len(X))
        for estimator in range(0, self.n_estimators):
            predictions[estimator] = self.ensemble[estimator].predict(X)
        for i in range(0, len(X)):
            pred = {}
            for j in range(0, self.n_estimators):
                if predictions[j][i] in pred:
                    pred[predictions[j][i]] += self.ensemble[j].fitness
                else:
                    pred[predictions[j][i]]  = self.ensemble[j].fitness
            y[i] = max(pred.items(), key=operator.itemgetter(1))[0]
        return y
