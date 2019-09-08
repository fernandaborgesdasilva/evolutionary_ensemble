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
from parallel_brute_force_random_search import BruteForceEnsembleClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from joblib import Memory
memory = Memory("./tmpmemoryjoblib", verbose=0)
    
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
                'KNeighborsClassifier': {'n_neighbors':[1, 3, 7, n_samples], 'weights':['uniform', 'distance']},
                #RidgeClassifier(): {'alpha':[1.0, 10.0],'max_iter':[10, 100]},
                'SVC': {'C':[1, 1000],'gamma':[0.0001, 0.001]},
                'DecisionTreeClassifier': {'min_samples_leaf':[1, 5], 'max_depth':[None, 5]},
                #ExtraTreeClassifier(): {'min_samples_leaf':[1, n_samples], 'max_depth':[1, n_samples]},
                'GaussianNB': {},
                'LinearDiscriminantAnalysis': {},
                #QuadraticDiscriminantAnalysis(): {},
                #BernoulliNB(): {},
                'LogisticRegression': {'C':[1, 1000], 'max_iter':[100]},
                #NearestCentroid(): {},
                'PassiveAggressiveClassifier': {'C':[1, 1000], 'max_iter':[100]},
                'SGDClassifier': {'alpha':[1e-5, 1e-2], 'max_iter':[100]}
    }
    all_ensembles = []
    for i, classifiers in enumerate(combinations(estimators(alg),n_estimators)):
        all_ensembles.append([classifiers])
    return all_ensembles

def compare_results(data, target, n_estimators, outputfile, stop_time, all_possible_ensembles, possible_ensembles_time, n_cores):
    accuracy, f1, precision, recall, auc = 0, 0, 0, 0, 0
    total_accuracy, total_f1, total_precision, total_recall, total_auc = 0, 0, 0, 0, 0
    with open(outputfile, "w") as text_file:
        text_file.write('*'*60)
        text_file.write(' Brute Force Ensemble Classifier  - Parallel version')
        text_file.write('*'*60)
        text_file.write('\nAll possible ensembles combinations created in %i' % (possible_ensembles_time))
        text_file.write(" ms.")
        text_file.write('\n\nn_estimators = %i' % (n_estimators))
        text_file.write('\nstop_time = %i' % (stop_time))
        for i in range(0, 10):
            csv_file = 'pbfec_rand_results_iter_' + str(i) + '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
            ensemble_classifier = BruteForceEnsembleClassifier(stop_time=stop_time, n_estimators=int(n_estimators), random_state=i*10)
            print('\n\nIteration = ',i)
            text_file.write("\n\nIteration = %i" % (i))
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=i*10)
            selected_ensemble = random.sample(range(len(all_possible_ensembles)), k=stop_time)
            fit_aux = int(round(time.time() * 1000))
            search_results = ensemble_classifier.fit(X_train, y_train, all_possible_ensembles, selected_ensemble, n_cores)
            #saving results as pandas dataframe and csv
            search_results_pd = pd.DataFrame(search_results)
            search_results_pd.to_csv(csv_file, index = None, header=True)     
            ensemble = search_results_pd.loc[search_results_pd['fitness_ensemble'].idxmax()]["ensemble"]
            best_fitness_classifiers = search_results_pd.loc[search_results_pd['fitness_ensemble'].idxmax()]["fitness_classifiers"]
            ensemble_classifier.fit_ensemble(X_train, y_train, ensemble[0], best_fitness_classifiers)
            fit_total_time = (int(round(time.time() * 1000)) - fit_aux)
            text_file.write("\n\nBFEC fit done in %i" % (fit_total_time))
            text_file.write(" ms")
            predict_aux = int(round(time.time() * 1000))
            y_pred = ensemble_classifier.predict(X_test)
            predict_total_time = (int(round(time.time() * 1000)) - predict_aux)
            text_file.write("\n\nBFEC predict done in %i" % (predict_total_time))
            text_file.write(" ms")
            accuracy = accuracy_score(y_test, y_pred)
            total_accuracy += accuracy
            try: 
                f1 = f1_score(y_test, y_pred)
                total_f1 += f1
            except: pass
            try: 
                precision = precision_score(y_test, y_pred)
                total_precision += precision
            except: pass
            try: 
                recall = recall_score(y_test, y_pred)
                total_recall += recall
            except: pass
            try: 
                auc = roc_auc_score(y_test, y_pred)
                total_auc += auc
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
            memory.clear(warn=False)    
        text_file.write("\n\nAverage Accuracy = %f\n" % (total_accuracy/10))
        if total_f1>0:
            text_file.write("Average F1-score = %f\n" % (total_f1/10))
        if total_precision>0:
            text_file.write("Average Precision = %f\n" % (total_precision/10))
        if total_recall>0:
            text_file.write("Average Recall = %f\n" % (total_recall/10))
        if total_auc>0:
            text_file.write("Average ROC AUC = %f\n" % (total_auc/10))
            
def main(argv):
    inputfile = ''
    outputfile = ''
    n_estimators = ''
    stop_time = ''
    try:
        opts, args = getopt.getopt(argv,"h:i:o:e:s:c:",["ifile=","ofile=","enumber=","stoptime=","cores="])
    except getopt.GetoptError:
        print('parallel_brute_force_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time>  -c <n_cores>')
        sys.exit(2)
    if opts == []:
        print('parallel_brute_force_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('parallel_brute_force_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-e", "--enumber"):
            n_estimators = arg
        elif opt in ("-s", "--stoptime"):
            stop_time = arg
        elif opt in ("-c", "--cores"):
            n_cores = arg
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    print('The number of estimators is ', n_estimators)
    print('The number of iterations is ', stop_time)
    print('The number of cores is ', n_cores)
    if inputfile == "iris":
        dataset = datasets.load_iris()
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.data, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinations created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier (parallel version)...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time), 
                        all_possible_ensembles=possible_ensembles,
                        possible_ensembles_time=total_time,
                        n_cores=int(n_cores)
                       )
    elif inputfile == "breast":
        dataset = datasets.load_breast_cancer()
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.data, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinations created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier (parallel version)...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time), 
                        all_possible_ensembles=possible_ensembles,
                        possible_ensembles_time=total_time,
                        n_cores=int(n_cores)
                       )
    elif  inputfile == "wine":
        dataset = datasets.load_wine()
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.data, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinations created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier (parallel version)...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time), 
                        all_possible_ensembles=possible_ensembles,
                        possible_ensembles_time=total_time,
                        n_cores=int(n_cores)
                       )
    else:
        le = LabelEncoder()
        dataset = pd.read_csv(inputfile)
        dataset.iloc[:, -1] = le.fit_transform(dataset.iloc[:, -1])
        aux = int(round(time.time() * 1000))
        possible_ensembles = define_all_possible_ensembles(data=dataset.iloc[:, 0:-1].values, n_estimators=int(n_estimators))
        total_time = (int(round(time.time() * 1000)) - aux)
        print("\nAll possible ensembles combinations created in ", total_time, " ms.")
        print('Runing Brute Force Ensemble Classifier (parallel version)...')
        compare_results(data=dataset.iloc[:, 0:-1].values, 
                        target=dataset.iloc[:, -1].values, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time), 
                        all_possible_ensembles=possible_ensembles,
                        possible_ensembles_time=total_time,
                        n_cores=int(n_cores)
                       )
    print('It is finished!')

if __name__ == "__main__":
    main(sys.argv[1:])