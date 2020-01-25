from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import random
import operator
import time
import sys, getopt
import copy
from joblib import Parallel, delayed, load, dump
from all_members_ensemble import gen_members

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Chromossome:
    def __init__(self, genotypes_pool, random_state=None):
        self.genotypes_pool = genotypes_pool
        self.classifier = None
        self.classifier_algorithm = None
        self.mutate()
        self.fitness = 0
        self.random_state = random_state

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def mutate(self, n_positions=None):
        change_classifier = random.randint(0, len(self.genotypes_pool))
        if self.classifier is None or change_classifier != 0:
            param = {}
            self.classifier_algorithm = random.choice(list(self.genotypes_pool.keys()))
            mod, f = self.classifier_algorithm.rsplit('.', 1)
            clf = getattr(__import__(mod, fromlist=[f]), f)()
        else:
            param = self.classifier.get_params()
            clf = self.classifier

        if not n_positions or n_positions>len(self.genotypes_pool[self.classifier_algorithm]):
            n_positions = len(self.genotypes_pool[self.classifier_algorithm])

        mutation_positions = random.sample(range(0, len(self.genotypes_pool[self.classifier_algorithm])), n_positions)
        i=0
        for hyperparameter, h_range in self.genotypes_pool[self.classifier_algorithm].items():
            if i in mutation_positions or self.classifier_algorithm != self.classifier.__class__:
                if isinstance(h_range[0], str):
                    param[hyperparameter] = random.choice(h_range)
                elif isinstance(h_range[0], float):
                    h_range_ = []
                    h_range_.append(min(h_range))
                    h_range_.append(max(h_range))
                    param[hyperparameter] = random.uniform(h_range_[0], h_range_[1]+1)
                else:
                    h_range_ = []
                    h_range_.append(min(h_range))
                    h_range_.append(max(h_range))
                    param[hyperparameter] = random.randint(h_range_[0], h_range_[1]+1)
            i+= 1

        self.classifier = clf.set_params(**param)

        try:
            self.classifier.set_param(random_state=self.random_state)
        except:
            pass
        
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

class DiversityEnsembleClassifier:
    def __init__(self, algorithms, population_size = 100, max_epochs = 100, random_state=None):
        self.population_size = population_size
        self.max_epochs = max_epochs
        self.population = []
        self.algorithms = algorithms
        self.random_state = random_state
        self.ensemble = []
        random.seed(self.random_state)
        for i in range(0, population_size):
            self.population.append(Chromossome(genotypes_pool=algorithms, random_state=random_state))

    def generate_offspring(self, parents, children, pop_fitness):
        children_aux = children
        if not parents:
            parents = [x for x in range(0, self.population_size)]
            children = [x for x in range(self.population_size, 2*self.population_size)]
        
        if (len(parents) < self.population_size):
            diff = self.population_size - len(parents)
            for i in range(0, diff):
                max_fit_index_aux = np.argmax(pop_fitness)
                self.population[children_aux[i]] = copy.deepcopy(self.population[max_fit_index_aux])
                parents.append(children_aux[i])
                pop_fitness = np.delete(pop_fitness, max_fit_index_aux)
                children = np.delete(children, i)
                
        for i in range(0, self.population_size):
            new_chromossome = copy.deepcopy(self.population[parents[i]])
            new_chromossome.mutate()
            try:
                self.population[children[i]] = new_chromossome
            except:
                self.population.append(new_chromossome)

    def fit_predict_population(self, not_fitted, kfolds, X, y):
        predictions = np.empty([y.shape[0]])
        chromossome = self.population[not_fitted]
        for train, val in kfolds.split(X):
            chromossome.fit(X[train], y[train])
            predictions[val] = np.equal(chromossome.predict(X[val]), y[val])
        return [not_fitted, predictions]

    def diversity_selection(self, predictions, selection_threshold):
        #print(-1)
        distances = np.zeros(2*self.population_size)
        pop_fitness = predictions.sum(axis=1)/predictions.shape[1]
        target_chromossome = np.argmax(pop_fitness)
        selected = [target_chromossome]
        self.population[target_chromossome].fitness = pop_fitness[target_chromossome]
        diversity  = np.zeros(2*self.population_size)
        mean_fitness = pop_fitness[target_chromossome]
        distances[pop_fitness < selection_threshold] = float('-inf')
        for i in range(0, self.population_size-1):
            distances[target_chromossome] = float('-inf')
            d_i = np.logical_xor(predictions, predictions[target_chromossome]).sum(axis=1)
            distances += d_i
            target_chromossome = np.argmax(distances)
            if distances[target_chromossome] == float('-inf'):
                break
            diversity += d_i/predictions.shape[1]
            mean_fitness += pop_fitness[target_chromossome]
            selected.append(target_chromossome)
            self.population[target_chromossome].fitness = pop_fitness[target_chromossome]

        return selected, (diversity[selected]/self.population_size).mean(), mean_fitness/(self.population_size), pop_fitness

    def fit(self, X, y, n_cores):
        diversity_values, fitness_values = [], []
        result_dict = dict()
        ##print('Starting genetic algorithm...')
        kf = KFold(n_splits=5, random_state=self.random_state)
        start_time = int(round(time.time() * 1000))
        random.seed(self.random_state)

        selected, not_selected = [], []
        selected, not_selected, pop_fitness = [], [], []
        all_predictions = np.zeros([2*self.population_size, y.shape[0]])

        frequencies = np.unique(y, return_counts=True)[1]
        selection_threshold = max(frequencies)/np.sum(frequencies)

        for epoch in range(self.max_epochs):
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            time_aux = int(round(now * 1000))
            
            ensemble = []
            classifiers_fitness = []

            not_selected = np.setdiff1d([x for x in range(0, 2*self.population_size)], selected)
            self.generate_offspring(selected, not_selected, pop_fitness)
            
            backend = 'loky'
            fit_predictions = Parallel(n_jobs=n_cores, backend=backend)(delayed(self.fit_predict_population)(item, kf, X, y) for item in not_selected)
            
            for i in fit_predictions:
                all_predictions[i[0]] = i[1]

            #print('Applying diversity selection...', end='')
            selected, diversity, fitness, pop_fitness = self.diversity_selection(all_predictions, selection_threshold)
            diversity_values.append(diversity)
            fitness_values.append(fitness)
            for chromossome in self.population:
                ensemble.append(chromossome.classifier)
                classifiers_fitness.append(chromossome.fitness)
                
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            end_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            total_time = (int(round(now * 1000)) - time_aux)
            result_dict.update({epoch:{"start_time":start_time,
                                       "end_time":end_time,
                                       "total_time_ms":total_time,
                                       "diversity":diversity,
                                       "fitness":fitness,
                                       "ensemble":ensemble, 
                                       "classifiers_fitness":classifiers_fitness}})
        return result_dict
    
    def fit_ensemble(self, X, y, ensemble, classifiers_fitness):
        classifiers_fitness_it = 0
        for estimator in ensemble:
            self.ensemble.append(Estimator(classifier=estimator, random_state=self.random_state, fitness=classifiers_fitness[classifiers_fitness_it]))
            classifiers_fitness_it = classifiers_fitness_it + 1
        for classifier in self.ensemble:
            classifier.fit(X, y)

    def predict(self, X):
        len_X = len(X)
        predictions = np.zeros((self.population_size, len_X))
        y = np.zeros(len_X)
        for chromossome in range(0, self.population_size):
            predictions[chromossome] = self.ensemble[chromossome].predict(X)
        for i in range(0, len_X):
            pred = {}
            for j in range(0, self.population_size):
                if predictions[j][i] in pred:
                    pred[predictions[j][i]] += self.ensemble[j].fitness
                else:
                    pred[predictions[j][i]]  = self.ensemble[j].fitness
            y[i] = max(pred.items(), key=operator.itemgetter(1))[0]
        return y
    
def compare_results(data, target, n_estimators, outputfile, stop_time, n_cores):
    accuracy, f1, precision, recall, auc = 0, 0, 0, 0, 0
    total_accuracy, total_f1, total_precision, total_recall, total_auc = 0, 0, 0, 0, 0
    div = np.zeros(stop_time)
    fit = np.zeros(stop_time)
    fit_total_time = 0
    alg = gen_members(data.shape)
    
    with open(outputfile, "w") as text_file:
        text_file.write('*'*60)
        text_file.write(' Diversity-based Ensemble Classifier ')
        text_file.write('*'*60)
        text_file.write('\n\nn_estimators = %i' % (n_estimators))
        text_file.write('\nstop_time = %i' % (stop_time))
        for i in range(0, 10):
            csv_file = 'parallel_diversity_results_iter_' + str(i) + '_' + str(n_cores) + '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=i*10)
            ensemble_classifier = DiversityEnsembleClassifier(algorithms=alg, 
                                                              population_size=n_estimators, 
                                                              max_epochs=stop_time, random_state=i*10)
            print('\n\nIteration = ',i)
            text_file.write("\n\nIteration = %i" % (i))
            fit_aux = int(round(time.time() * 1000))
            results = ensemble_classifier.fit(X_train, y_train, n_cores)
            results_pd = pd.DataFrame.from_dict(results, orient='index')
            results_pd.to_csv (csv_file, index = None, header=True)
            ensemble = results_pd[-1:]["ensemble"].item()
            classifiers_fitness = results_pd[-1:]["classifiers_fitness"].item()
            ensemble_classifier.fit_ensemble(X_train, y_train, ensemble, classifiers_fitness)
            fit_total_time = (int(round(time.time() * 1000)) - fit_aux)
            text_file.write("\n\nDEC fit done in %i" % (fit_total_time))
            text_file.write(" ms")
            predict_aux = int(round(time.time() * 1000))
            y_pred = ensemble_classifier.predict(X_test)
            predict_total_time = (int(round(time.time() * 1000)) - predict_aux)
            text_file.write("\n\nDEC predict done in %i" % (predict_total_time))
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
    save_results = 'diversity_ensemble_results_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + ".csv"
    try:
        opts, args = getopt.getopt(argv,"h:i:o:e:s:c:",["ifile=","ofile=","enumber=","stoptime=","cores="])
    except getopt.GetoptError:
        print('parallel_diversity_ensemble.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
        sys.exit(2)
    if opts == []:
        print('parallel_diversity_ensemble.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('parallel_diversity_ensemble.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores>')
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
        print('Runing Diversity-based Ensemble Classifier (parallel version)...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time),
                        n_cores=int(n_cores)
                       )
    elif inputfile == "breast":
        dataset = datasets.load_breast_cancer()
        print('Runing Diversity-based Ensemble Classifier (parallel version)...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time),
                        n_cores=int(n_cores)
                       )
    elif  inputfile == "wine":
        dataset = datasets.load_wine()
        print('Runing Diversity-based Ensemble Classifier (parallel version)...')
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
        print('Runing Diversity-based Ensemble Classifier (parallel version)...')
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