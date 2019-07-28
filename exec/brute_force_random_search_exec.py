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
    
def estimators(estimator_grid):
    for estimator, param_grid in list(estimator_grid.items()): 
        items = sorted(param_grid.items())
        if not items:
            yield (estimator, {})
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield (estimator, params)
                
def define_all_possible_ensembles(data, n_estimators=10):
    n_samples = int(math.sqrt(data.shape[0]))
    alg = {
                KNeighborsClassifier(): {'n_neighbors':[1, 3, 7, n_samples], 'weights':['uniform', 'distance']},
                #RidgeClassifier(): {'alpha':[1.0, 10.0],'max_iter':[10, 100]},
                SVC(): {'C':[1, 1000],'gamma':[0.0001, 0.001]},
                DecisionTreeClassifier(): {'min_samples_leaf':[1, 5], 'max_depth':[None, 5]},
                #ExtraTreeClassifier(): {'min_samples_leaf':[1, n_samples], 'max_depth':[1, n_samples]},
                GaussianNB(): {},
                LinearDiscriminantAnalysis(): {},
                #QuadraticDiscriminantAnalysis(): {},
                #BernoulliNB(): {},
                LogisticRegression(): {'C':[1, 1000], 'max_iter':[100]},
                #NearestCentroid(): {},
                PassiveAggressiveClassifier(): {'C':[1, 1000], 'max_iter':[100]},
                SGDClassifier(): {'alpha':[1e-5, 1e-2], 'max_iter':[100]}
    }
    all_ensembles = []
    for i, classifiers in enumerate(combinations(estimators(alg),n_estimators)):
        all_ensembles.append([classifiers])
    return all_ensembles

def compare_results(data, target, n_estimators, csv_file, outputfile, stop_time, all_possible_ensembles):
    accuracy, f1, precision, recall, auc = 0, 0, 0, 0, 0
    with open(outputfile, "w") as text_file:
        text_file.write('*'*60)
        text_file.write(' Brute Force Ensemble Classifier ')
        text_file.write('*'*60)
        text_file.write('\n\nn_estimators = %i' % (n_estimators))
        text_file.write('\nstop_time = %i' % (stop_time))
        ensemble_classifier = BruteForceEnsembleClassifier(stop_time=stop_time, n_estimators=int(n_estimators), random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
        fit_aux = int(round(time.time() * 1000))
        search_results = ensemble_classifier.fit(X_train, y_train, all_possible_ensembles)
        #saving results as pandas dataframe and csv
        search_results_pd = pd.DataFrame.from_dict(search_results, orient='index')
        search_results_pd.to_csv (csv_file, index = None, header=True)       
        ensemble = search_results_pd[-1:]["ensemble"].item()
        best_fitness_classifiers = search_results_pd[-1:]["best_fitness_classifiers"].item()
        ensemble_classifier.fit_ensemble(X_train, y_train, ensemble[0], best_fitness_classifiers)
        fit_total_time = (int(round(time.time() * 1000)) - fit_aux)
        text_file.write("\n\nBFEC fit done in %i" % (fit_total_time))
        text_file.write(" ms")
        predict_aux = int(round(time.time() * 1000))
        y_pred = ensemble_classifier.predict(X_test)
        predict_total_time = (int(round(time.time() * 1000)) - predict_aux)
        text_file.write("\n\nBFEC predict done in %i" % (predict_total_time))
        text_file.write(" ms")
        accuracy += accuracy_score(y_test, y_pred)
        try: f1 += f1_score(y_test, y_pred)
        except: pass
        try: precision += precision_score(y_test, y_pred)
        except: pass
        try: recall += recall_score(y_test, y_pred)
        except: pass
        try: auc += roc_auc_score(y_test, y_pred)
        except: pass
        text_file.write("\n\nAccuracy = %f\n" % (accuracy))
        if f1>0:
            text_file.write("F1-score = %f\n" % (f1))
        if precision>0:
            text_file.write("Precision = %f\n" % (precision))
        if recall>0:
            text_file.write("Recall = %f\n" % (recall))
        if auc>0:
            text_file.write("ROC AUC = %f\n" % (auc))
            
def main(argv):
    inputfile = ''
    outputfile = ''
    n_estimators = ''
    stop_time = ''
    save_results = 'brute_force_random_search_results_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + ".csv"
    try:
        opts, args = getopt.getopt(argv,"h:i:o:e:s:",["ifile=","ofile=","enumber=","stoptime="])
    except getopt.GetoptError:
        print('brute_force_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time>')
        sys.exit(2)
    if opts == []:
        print('brute_force_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('brute_force_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-e", "--enumber"):
            n_estimators = arg
        elif opt in ("-s", "--stoptime"):
            stop_time = arg
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    print('The number of estimators is ', n_estimators)
    print('The number of iterations is ', stop_time)
    if inputfile == "iris":
        dataset = datasets.load_iris()
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.data, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinatios created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier...')
        compare_results(data=dataset.data, target=dataset.target, n_estimators=int(n_estimators), csv_file=save_results, outputfile=outputfile, stop_time=int(stop_time), all_possible_ensembles=possible_ensembles)
    elif inputfile == "breast":
        dataset = datasets.load_breast_cancer()
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.data, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinatios created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier...')
        compare_results(data=dataset.data, target=dataset.target, n_estimators=int(n_estimators), csv_file=save_results, outputfile=outputfile, stop_time=int(stop_time), all_possible_ensembles=possible_ensembles)
    elif  inputfile == "wine":
        dataset = datasets.load_wine()
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.data, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinatios created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier...')
        compare_results(data=dataset.data, target=dataset.target, n_estimators=int(n_estimators), csv_file=save_results, outputfile=outputfile, stop_time=int(stop_time), all_possible_ensembles=possible_ensembles)
    else:
        le = LabelEncoder()
        dataset = pd.read_csv(inputfile)
        dataset.iloc[:, -1] = le.fit_transform(dataset.iloc[:, -1])
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.iloc[:, 0:-1].values, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinatios created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier...')
        compare_results(data=dataset.iloc[:, 0:-1].values, target=dataset.iloc[:, -1].values, n_estimators=int(n_estimators), csv_file=save_results, outputfile=outputfile, stop_time=int(stop_time), all_possible_ensembles=possible_ensembles)
    print('It is finished!')

if __name__ == "__main__":
    main(sys.argv[1:])
