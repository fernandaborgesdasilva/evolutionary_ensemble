from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import random
from itertools import product, combinations
import pandas as pd
import sys, getopt
import shutil
from sklearn.model_selection import KFold
import numpy as np
from scipy import stats
import operator
import time
from joblib import Parallel, delayed, load, dump
from all_members_ensemble import gen_members

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from joblib import Memory
cachedir = './parallel_brute_force_random_search_exec_tmpmemory' + '_' + time.strftime("%H_%M_%S", time.localtime(time.time()))
memory = Memory(cachedir, verbose=0)

@memory.cache
def train_clf(classifier, params, X, y, random_state):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    #clf = getattr(sys.modules[__name__], classifier)()
    mod, f = classifier.rsplit('.', 1)
    clf = getattr(__import__(mod, fromlist=[f]), f)()
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
            #estimator = getattr(sys.modules[__name__], estimator)()
            mod, f = estimator.rsplit('.', 1)
            estimator =  getattr(__import__(mod, fromlist=[f]), f)()
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
                
def define_all_possible_ensembles(data, n_estimators):
    alg = gen_members(data.shape)
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
            csv_file = 'pbfec_rand_results_iter_' + str(i) + '_' + str(n_cores) + '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
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
            shutil.rmtree(cachedir)
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