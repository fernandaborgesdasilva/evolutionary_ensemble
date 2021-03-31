from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import statistics
import operator
import time
import sys, getopt, os
import shutil
from all_members_ensemble import gen_members
from joblib import Parallel, delayed
import asyncio
from aiofile import AIOFile, Writer
import nest_asyncio
from numpy.random import RandomState
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

from joblib import Memory
cachedir = './random_search_exec_tmpmemory' + '_' + time.strftime("%H_%M_%S", time.localtime(time.time()))
memory = Memory(cachedir, verbose=0)

class Estimator:
    def __init__(self, classifier=None, random_state=None, accuracy=0):
        self.classifier = classifier
        self.random_state = random_state
        self.accuracy = accuracy

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)


class BruteForceEnsembleClassifier:
    def __init__(self, classifiers_pool, stop_time = 100, n_estimators = 10, random_state = None):
        self.n_estimators = n_estimators
        self.ensemble = []
        self.stop_time = stop_time
        self.random_state = random_state
        self.classifiers_pool = classifiers_pool

    def get_new_classifier(self, random_id):
        self.rnd = RandomState(random_id)
        param = {}
        classifier_algorithm = list(self.classifiers_pool.keys())[self.rnd.choice(len(list(self.classifiers_pool.keys())))]
        mod, f = classifier_algorithm.rsplit('.', 1)
        clf = getattr(__import__(mod, fromlist=[f]), f)()
        for hyperparameter, h_range in self.classifiers_pool[classifier_algorithm].items():
            if isinstance(h_range[0], str):
                    param[hyperparameter] = h_range[self.rnd.choice(len(h_range))]
            elif isinstance(h_range[0], float):
                h_range_ = []
                h_range_.append(min(h_range))
                h_range_.append(max(h_range))
                param[hyperparameter] = self.rnd.uniform(h_range_[0], h_range_[1]+1)
            else:
                h_range_ = []
                h_range_.append(min(h_range))
                h_range_.append(max(h_range))
                param[hyperparameter] = self.rnd.randint(h_range_[0], h_range_[1]+1)
        classifier = clf.set_params(**param)
        all_parameters = classifier.get_params()
        if 'random_state' in list(all_parameters.keys()):
            classifier.set_params(random_state=self.random_state)
        return classifier

    def parallel_fit(self, x_file, y_file, len_X, len_y, y_true, classifiers):
        now = time.time()
        struct_now = time.localtime(now)
        mlsec = repr(now).split('.')[1][:3]
        start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
        time_aux = int(round(now * 1000))
        result_dict = dict()
        # a matrix with all observations vs the prediction of each classifier
        classifiers_predictions = np.zeros([self.n_estimators, len_y])
        # sum the number of right predictions for each classifier
        classifiers_right_predictions = np.zeros([self.n_estimators])
        ensemble_accuracy = np.zeros([len_y])
        classifier_id = 0
        for classifier in classifiers:
            y_pred = train_clf(classifier, x_file, y_file, self.random_state)
            classifiers_predictions[classifier_id][:] = y_pred
            classifiers_right_predictions[classifier_id] = accuracy_score(y_true, y_pred)
            classifier_id = classifier_id + 1

        y_train_pred = np.zeros(len_X)
        for i in range(0, len_X):
            pred = {}
            for j in range(0,self.n_estimators):
                if classifiers_predictions[j][i] in pred:
                    pred[classifiers_predictions[j][i]] += classifiers_right_predictions[j]
                else:
                    pred[classifiers_predictions[j][i]]  = classifiers_right_predictions[j]
            y_train_pred[i] = max(pred.items(), key=operator.itemgetter(1))[0]

            ensemble_accuracy = accuracy_score(y_true, y_train_pred)

        now = time.time()
        struct_now = time.localtime(now)
        mlsec = repr(now).split('.')[1][:3]
        end_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
        total_time = (int(round(now * 1000)) - time_aux)
        result_dict.update({"start_time":start_time,
                            "end_time":end_time,
                            "total_time_ms":total_time,
                            "ensemble_accuracy":ensemble_accuracy,
                            "ensemble":classifiers, 
                            "accuracy_classifiers":list(classifiers_right_predictions)})
        return result_dict

    def fit(self, X, y, n_cores, csv_file, random_state):
        #parallel_time_aux = int(round(time.time() * 1000))
        #x_train_file_path = "/dev/shm/temp_x_train_prs" + str(n_cores) + ".npy"
        #y_train_file_path = "/dev/shm/temp_y_train_prs" + str(n_cores) + ".npy"
        x_train_file_path = "./temp_x_train_prs.npy"
        y_train_file_path = "./temp_y_train_prs.npy"
        np.save(x_train_file_path, X)
        np.save(y_train_file_path, y)
        len_y = len(y)
        len_X = len(X)
        result = []
        random_id = random_state
        num_interations = int(math.ceil(self.stop_time/n_cores))
        for index in range(num_interations):
            selected_ensembles = []
            for ens in range(0, n_cores):
                ensemble = []
                for cl in range(0, self.n_estimators):
                    random_id = random_id + 1
                    classifier = self.get_new_classifier(random_id)
                    ensemble.append(classifier)
                selected_ensembles.append(ensemble)
            backend = 'loky'
            result.extend(Parallel(n_jobs=n_cores, backend=backend)(delayed(self.parallel_fit)(x_train_file_path, 
                                                                                          y_train_file_path,
                                                                                          len_X,
                                                                                          len_y,
                                                                                          y,
                                                                                          item) for item in selected_ensembles))
        #total_parallel_time = (int(round(time.time() * 1000)) - parallel_time_aux)
        #print("\n>>>>> Parallel step processing time = %i" % (total_parallel_time))
        os.remove(x_train_file_path)
        os.remove(y_train_file_path)
        return result

    def fit_ensemble(self, X, y, ensemble, best_accuracy_classifiers):
        classifiers_accuracy_it = 0
        for estimator in ensemble:
            self.ensemble.append(Estimator(classifier=estimator, 
                                           random_state=self.random_state, 
                                           accuracy=best_accuracy_classifiers[classifiers_accuracy_it]))
            classifiers_accuracy_it = classifiers_accuracy_it + 1
        for classifier in self.ensemble:
            classifier.fit(X, y)

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
                    pred[predictions[j][i]] += self.ensemble[j].accuracy
                else:
                    pred[predictions[j][i]]  = self.ensemble[j].accuracy
            y[i] = max(pred.items(), key=operator.itemgetter(1))[0]
        return y

@memory.cache
def train_clf(classifier, x_file, y_file, random_state):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    X = np.load(x_file)
    y = np.load(y_file)
    y_pred = np.zeros([len(y)])
    #k-fold cross-validation
    kf = KFold(n_splits=5, random_state=random_state)
    for train, val in kf.split(X):
        classifier.fit(X[train], y[train])
        y_pred[val] = classifier.predict(X[val])
    return y_pred

async def writing_results_task(result_dict, csv_file):
    async with AIOFile(csv_file, 'a') as afp:
        writer = Writer(afp)
        await writer(pd.DataFrame.from_dict(result_dict, orient='index').to_csv(header=False, index=None))
        await afp.fsync()  

def compare_results(data, target, n_estimators, outputfile, stop_time, n_cores):
    accuracy, f1, precision, recall, auc = 0, 0, 0, 0, 0
    total_accuracy, total_f1, total_precision, total_recall, total_auc = [], [], [], [], []
    sum_total_iter_time = []
    alg = gen_members(data.shape)
    with open(outputfile, "w") as text_file:
        text_file.write('*'*60)
        text_file.write(' Random Search Ensemble Classifier ')
        text_file.write('*'*60)
        text_file.write('\n\nn_estimators = %i' % (n_estimators))
        text_file.write('\nstop_time = %i' % (stop_time))
        fold = 0
        kf = KFold(n_splits=5, random_state=42)
        for train, val in kf.split(data):
            print('\n\n>>>>>>>>>> Fold = ',fold)
            text_file.write("\n\n>>>>>>>>>> Fold = %i" % (fold))
            for i in range(0, 10):
                fit_time_aux = int(round(time.time() * 1000))
                csv_file = 'parallel_rs_results_fold_' + str(fold) + '_iter_' + str(i) + '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
                print('\n\nIteration = ',i)
                text_file.write("\n\nIteration = %i" % (i))
                ensemble_classifier = BruteForceEnsembleClassifier(classifiers_pool = alg,
                                                                   stop_time=stop_time, 
                                                                   n_estimators=int(n_estimators), 
                                                                   random_state=i*10)
                search_results = ensemble_classifier.fit(data[train],target[train], n_cores, csv_file, i*10)
                search_results_pd = pd.DataFrame(search_results)
                search_results_pd.to_csv(csv_file, index = None, header=True) 
                ensemble = search_results_pd.loc[search_results_pd['ensemble_accuracy'].idxmax()]["ensemble"]
                best_accuracy_classifiers = search_results_pd.loc[search_results_pd['ensemble_accuracy'].idxmax()]["accuracy_classifiers"]
                ensemble_classifier.fit_ensemble(data[train], target[train], ensemble, best_accuracy_classifiers)
                fit_total_time = (int(round(time.time() * 1000)) - fit_time_aux)
                text_file.write("\n\nParallel Random Search fit done in %i" % (fit_total_time))
                text_file.write(" ms")
                predict_aux = int(round(time.time() * 1000))
                y_pred = ensemble_classifier.predict(data[val])
                predict_total_time = (int(round(time.time() * 1000)) - predict_aux)
                text_file.write("\n\nParallel Random Search predict done in %i" % (predict_total_time))
                text_file.write(" ms")
                accuracy = accuracy_score(target[val], y_pred)
                total_accuracy.append(accuracy)
                try: 
                    f1 = f1_score(target[val], y_pred)
                    total_f1.append(f1)
                except: pass
                try: 
                    precision = precision_score(target[val], y_pred)
                    total_precision.append(precision)
                except: pass
                try: 
                    recall = recall_score(target[val], y_pred)
                    total_recall.append(recall)
                except: pass
                try: 
                    auc = roc_auc_score(target[val], y_pred)
                    total_auc.append(auc)
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
                total_iter_time = (int(round(time.time() * 1000)) - fit_time_aux)
                text_file.write("\nIteration done in %i" % (total_iter_time))
                text_file.write(" ms")
                sum_total_iter_time.append(total_iter_time)
            fold = fold + 1
        text_file.write("\n\nAverage Accuracy = %f\n" % (statistics.mean(total_accuracy)))
        text_file.write("Standard Deviation of Accuracy = %f\n" % (statistics.stdev(total_accuracy)))
        if sum(total_f1)>0:
            text_file.write("\nAverage F1-score = %f\n" % (statistics.mean(total_f1)))
            text_file.write("Standard Deviation of F1-score = %f\n" % (statistics.stdev(total_f1)))
        if sum(total_precision)>0:
            text_file.write("\nAverage Precision = %f\n" % (statistics.mean(total_precision)))
            text_file.write("Standard Deviation of Precision = %f\n" % (statistics.stdev(total_precision)))
        if sum(total_recall)>0:
            text_file.write("\nAverage Recall = %f\n" % (statistics.mean(total_recall)))
            text_file.write("Standard Deviation of Recall = %f\n" % (statistics.stdev(total_recall)))
        if sum(total_auc)>0:
            text_file.write("\nAverage ROC AUC = %f\n" % (statistics.mean(total_auc)))
            text_file.write("Standard Deviation of ROC AUC = %f\n" % (statistics.stdev(total_auc)))
        text_file.write("\n\nAverage duration of iterations = %i" % statistics.mean(sum_total_iter_time))
        text_file.write(" ms")
        text_file.write("\nStandard deviation of iterations duration = %i" % statistics.stdev(sum_total_iter_time))
        text_file.write(" ms\n")
            
def main(argv):
    inputfile = ''
    outputfile = ''
    n_estimators = ''
    stop_time = ''
    try:
        opts, args = getopt.getopt(argv,"h:i:o:e:s:c:",["ifile=","ofile=","enumber=","stoptime=","cores="])
    except getopt.GetoptError:
        print('parallel_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
        sys.exit(2)
    if opts == []:
        print('parallel_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('parallel_random_search_exec.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
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
        print('Runing Random Search Ensemble Classifier...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time),
                        n_cores=int(n_cores)
                       )
    elif inputfile == "breast":
        dataset = datasets.load_breast_cancer()
        print('Runing Random Search Ensemble Classifier...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time),
                        n_cores=int(n_cores)
                       )
    elif  inputfile == "wine":
        dataset = datasets.load_wine()
        print('Runing Random Search Ensemble Classifier...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time),
                        n_cores=int(n_cores)
                       )
    else:
        le = LabelEncoder()
        dataset = pd.read_csv(inputfile)
        dataset.iloc[:, -1] = le.fit_transform(dataset.iloc[:, -1])
        print('Runing Random Search Ensemble Classifier...')
        compare_results(data=dataset.iloc[:, 0:-1].values, 
                        target=dataset.iloc[:, -1].values, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time),
                        n_cores=int(n_cores)
                       )
    print('It is finished!')

if __name__ == "__main__":
    main(sys.argv[1:])
