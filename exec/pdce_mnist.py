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
from numpy.random import RandomState
import pandas as pd
import operator
import time
import sys, getopt, os
import copy
from collections import defaultdict
from scipy.stats import truncnorm
from joblib import Parallel, delayed
import statistics
from mnist_alg_pool import gen_members
import asyncio
from aiofile import AIOFile, Writer
import nest_asyncio
nest_asyncio.apply()

import warnings
import itertools
warnings.filterwarnings("ignore", category=DeprecationWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

class Chromossome:
    def __init__(self, genotypes_pool, x_n_cols, col_as_gene, guided_mutation, random_state=None):
        self.genotypes_pool = genotypes_pool
        self.classifier = None
        self.classifier_algorithm = None
        self.fitness = 0
        self.random_state = random_state
        self.col_as_gene = col_as_gene
        self.guided_mutation = guided_mutation
        self.cols = []
        self.x_n_cols = x_n_cols

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)
        
    def predict(self, X):
        return self.classifier.predict(X)

    def create_individual(self, rnd):
        param = {}
        #defining the classifier
        self.classifier_algorithm = list(self.genotypes_pool.keys())[rnd.choice(len(list(self.genotypes_pool.keys())))]
        mod, f = self.classifier_algorithm.rsplit('.', 1)
        clf = getattr(__import__(mod, fromlist=[f]), f)()
        #defining the hyperparameters conf
        for hyperparameter, h_range in self.genotypes_pool[self.classifier_algorithm].items():
            if isinstance(h_range[0], str):
                param[hyperparameter] = h_range[rnd.choice(len(h_range))]
            elif isinstance(h_range[0], float):
                h_range_ = []
                h_range_.append(min(h_range))
                h_range_.append(max(h_range))
                param[hyperparameter] = rnd.uniform(h_range_[0], h_range_[1])
            else:
                h_range_ = []
                h_range_.append(min(h_range))
                h_range_.append(max(h_range))
                param[hyperparameter] = rnd.randint(h_range_[0], h_range_[1]+1)
        self.classifier = clf.set_params(**param)
        all_parameters = self.classifier.get_params()
        if 'random_state' in list(all_parameters.keys()):
            self.classifier.set_params(random_state=self.random_state)
        if self.col_as_gene == 1:
            #defining which columns will be used
            n_cols = rnd.randint(1, self.x_n_cols + 1)
            self.cols = rnd.choice(self.x_n_cols, n_cols, replace=False)
    
    def choose_hyper_value(self, rnd, mutation_guided_before, hyper_values, hyper_proba_, h_range, param, hyperparameter):
        if self.guided_mutation == 0 or mutation_guided_before == 0 or len(hyper_values) == 0:
            #it does a search for new values
            if isinstance(h_range[0], str):
                param[hyperparameter] = h_range[rnd.choice(len(h_range))]
            elif isinstance(h_range[0], float):
                h_range_ = []
                h_range_.append(min(h_range))
                h_range_.append(max(h_range))
                param[hyperparameter] = rnd.uniform(h_range_[0], h_range_[1])
            else:
                h_range_ = []
                h_range_.append(min(h_range))
                h_range_.append(max(h_range))
                param[hyperparameter] = rnd.randint(h_range_[0], h_range_[1] + 1)
        else:
            #it chooses between the values already used
            if isinstance(h_range[0], str):
                param[hyperparameter] = rnd.choice(hyper_values, 1, p=hyper_proba_).item()
            elif isinstance(h_range[0], float):
                mu = float(rnd.choice(hyper_values, 1, p=hyper_proba_))
                sigma = (max(h_range) - min(h_range))/80
                a,b = (min(h_range)-mu)/sigma, (max(h_range)-mu)/sigma
                param[hyperparameter] = truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=self.random_state)
            else:
                r_proba_val = int(rnd.choice(hyper_values, 1, p=hyper_proba_))
                min_noise = max(min(h_range), r_proba_val - 2)
                max_noise = min(max(h_range), r_proba_val + 2)
                param[hyperparameter] = rnd.choice(list(set(range(min_noise, max_noise + 1)) - set([r_proba_val])), 1).item()
        return param

    def change_classifier(self, rnd, hyperparameter_proba, mutation_guided_before):
        param = {}
        self.classifier_algorithm = list(self.genotypes_pool.keys())[rnd.choice(len(list(self.genotypes_pool.keys())))]
        mod, f = self.classifier_algorithm.rsplit('.', 1)
        clf = getattr(__import__(mod, fromlist=[f]), f)()
        #defining the hyperparameters conf
        for hyperparameter, h_range in self.genotypes_pool[self.classifier_algorithm].items():
            if len(h_range) > 1:
                hyper_values = []
                hyper_proba = []
                hyper_proba_ = []
                if self.guided_mutation == 1:
                    if hyperparameter_proba.get(self.classifier_algorithm, 0) != 0:
                        if hyperparameter_proba[self.classifier_algorithm].get(hyperparameter, 0) != 0:
                            hyper_values = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["value"]
                            hyper_proba = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["probability"]
                    for fitness in hyper_proba:
                        if sum(hyper_proba) > 0:
                            hyper_proba_.append(fitness/sum(hyper_proba))
                        else:
                            hyper_proba_.append(1.0/len(hyper_proba))
                param = self.choose_hyper_value(rnd, mutation_guided_before, hyper_values, hyper_proba_, h_range, param, hyperparameter)
            elif len(h_range) == 1:
                param[hyperparameter] = h_range[0]
        self.classifier = clf.set_params(**param)
        all_parameters = self.classifier.get_params()
        if 'random_state' in list(all_parameters.keys()):
            self.classifier.set_params(random_state=self.random_state)

    def change_hyperparameters(self, rnd, hyperparameter_proba, mutation_guided_before, n_positions):
        param = self.classifier.get_params()
        clf = self.classifier
        len_hyper = len(self.genotypes_pool[self.classifier_algorithm])
        #changing the hyperparameters conf
        if len_hyper != 0:
            possible_hypers = []
            for hyper, values in self.genotypes_pool[self.classifier_algorithm].items():
                if len(values) > 1:
                    possible_hypers.append(hyper)
            if len(possible_hypers) > 0:
                if len(possible_hypers) == 1:
                    n_positions = 1
                else:
                    n_positions = rnd.randint(1, len(possible_hypers) + 1)
                mutation_positions = rnd.choice(range(0, len(possible_hypers)), n_positions)
                for hyper_id in mutation_positions:
                    hyperparameter = possible_hypers[hyper_id]
                    h_range = self.genotypes_pool[self.classifier_algorithm][hyperparameter]
                    hyper_values = []
                    hyper_proba = []
                    hyper_proba_ = []
                    if self.guided_mutation == 1:
                        if hyperparameter_proba.get(self.classifier_algorithm, 0) != 0:
                            if hyperparameter_proba[self.classifier_algorithm].get(hyperparameter, 0) != 0:
                                hyper_values = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["value"]
                                hyper_proba = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["probability"]
                        for fitness in hyper_proba:
                            if sum(hyper_proba) > 0:
                                hyper_proba_.append(fitness/sum(hyper_proba))
                            else:
                                hyper_proba_.append(1.0/len(hyper_proba))
                    param = self.choose_hyper_value(rnd, mutation_guided_before, hyper_values, hyper_proba_, h_range, param, hyperparameter)
            self.classifier = clf.set_params(**param)
            all_parameters = self.classifier.get_params()
            if 'random_state' in list(all_parameters.keys()):
                self.classifier.set_params(random_state=self.random_state)

    def change_columns(self, rnd, columns_proba, mutation_guided_before):
        #changing the columns
        if self.guided_mutation == 1:
            cols_proba_ = []
            cols_values = columns_proba["value"]
            cols_proba = columns_proba["probability"]
            for fitness in cols_proba:
                if sum(cols_proba) > 0:
                    cols_proba_.append(fitness/sum(cols_proba))
                else:
                    cols_proba_.append(1.0/len(cols_proba))
            n_cols = rnd.randint(1, self.x_n_cols + 1)
            if mutation_guided_before == 0 or n_cols > (len(cols_values) - cols_proba_.count(0.0)):
                self.cols = rnd.choice(self.x_n_cols, n_cols, replace=False)
            else:
                self.cols = rnd.choice(cols_values, n_cols, p=cols_proba_, replace=False)
        else:
            n_cols = rnd.randint(1, self.x_n_cols + 1)
            self.cols = rnd.choice(self.x_n_cols, n_cols, replace=False)

    def mutate(self, rnd, hyperparameter_proba, columns_proba, mutation_guided_before=0, n_positions=None):
        if self.col_as_gene == 1:
            change = rnd.randint(0, 3)
        else:
            change = rnd.randint(0, 2)
        if self.classifier is None:
            self.create_individual(rnd)
        elif change == 0:
            self.change_classifier(rnd, hyperparameter_proba, mutation_guided_before)
        elif change == 1:
            self.change_hyperparameters(rnd, hyperparameter_proba, mutation_guided_before, n_positions)
        elif change == 2:
            self.change_columns(rnd, columns_proba, mutation_guided_before)
        
class Estimator:
    def __init__(self, classifier, random_state, fitness, ensemble_cols):
        self.classifier = classifier
        self.fitness = fitness
        self.cols = ensemble_cols

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

class DiversityEnsembleClassifier:
    def __init__(self, algorithms, x_n_cols, col_as_gene, guided_mutation, population_size = 100, max_epochs = 100, random_state=None):
        self.population_size = population_size
        self.max_epochs = max_epochs
        self.population = []
        self.algorithms = algorithms
        self.x_n_cols = x_n_cols
        self.random_state = random_state
        self.rnd = RandomState(self.random_state)
        self.col_as_gene = col_as_gene
        self.guided_mutation = guided_mutation
        self.ensemble = []
        self.hyperparameter_proba = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.columns_proba = {"value":[],"probability":[]}
        for i in range(0, population_size):
            self.population.append(Chromossome(genotypes_pool=self.algorithms, x_n_cols=self.x_n_cols, col_as_gene=self.col_as_gene, guided_mutation=self.guided_mutation, random_state=self.random_state))
        for i in range(0, population_size):
            self.population[i].mutate(self.rnd, self.hyperparameter_proba, self.columns_proba)
            
    def generate_offspring(self, parents, children, pop_fitness, mutation_guided_before):
        children_aux = children
        if not parents:
            parents = [x for x in range(0, self.population_size)]
            children = [x for x in range(self.population_size, 2*self.population_size)]
                    
        for i in range(0, self.population_size):
            new_chromossome = copy.deepcopy(self.population[parents[i]])
            new_chromossome.mutate(self.rnd, self.hyperparameter_proba, self.columns_proba, mutation_guided_before)
            try:
                self.population[children[i]] = new_chromossome
            except:
                self.population.append(new_chromossome)
        if mutation_guided_before == 0:
            return 1
        else:
            return 0

    def fit_predict_population(self, not_fitted, kfolds, X, y):
        predictions = np.empty([y.shape[0]])
        y_train_pred = np.empty([y.shape[0]])
        chromossome = self.population[not_fitted]
        for train, val in kfolds.split(X):
            if self.col_as_gene == 1:
                chromossome.fit(X[train][:,chromossome.cols], y[train])
                y_train_pred[val] = chromossome.predict(X[val][:,chromossome.cols])
            else:
                chromossome.fit(X[train], y[train])
                y_train_pred[val] = chromossome.predict(X[val])
            predictions[val] = np.equal(y_train_pred[val], y[val])
        return [not_fitted, predictions, y_train_pred]

    def diversity_selection(self, predictions, selection_threshold):
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
    
    def hyperparameter_proba_update(self, chromossome):
        for hyper in chromossome.classifier.get_params():
            if hyper in self.algorithms[chromossome.classifier_algorithm].keys():
                if isinstance(chromossome.classifier.get_params()[hyper], float):
                    value = round(chromossome.classifier.get_params()[hyper], 3)
                else:
                    value = chromossome.classifier.get_params()[hyper]
                    assert isinstance(value, int) or isinstance(value, str), 'hyperparameter value of wrong type!'
                aux_hyperparameter_proba_hyper = self.hyperparameter_proba[chromossome.classifier_algorithm][hyper]
                if self.hyperparameter_proba[chromossome.classifier_algorithm].get(hyper, 0) != 0:
                #this hyperparameter already exists in aux_hyperparameter_proba
                    if value in aux_hyperparameter_proba_hyper['value']:
                    #this hyperparameter value already exists in aux_hyperparameter_proba
                        index = aux_hyperparameter_proba_hyper["value"].index(value)
                        aux_hyperparameter_proba_hyper["probability"][index] += chromossome.fitness
                    else:
                    #adding this hyperparameter value to aux_hyperparameter_proba
                        if len(aux_hyperparameter_proba_hyper["value"]) < 20:
                            aux_hyperparameter_proba_hyper["value"].append(value)
                            aux_hyperparameter_proba_hyper["probability"].append(chromossome.fitness)
                        else:
                            min_proba = min(aux_hyperparameter_proba_hyper["probability"])
                            min_proba_index = aux_hyperparameter_proba_hyper["probability"].index(min_proba)
                            del aux_hyperparameter_proba_hyper["probability"][min_proba_index]
                            del aux_hyperparameter_proba_hyper["value"][min_proba_index]
                            aux_hyperparameter_proba_hyper["value"].append(value)
                            aux_hyperparameter_proba_hyper["probability"].append(chromossome.fitness)
                else:
                #adding the hyperparameter to aux_hyperparameter_proba
                    aux_hyperparameter_proba_hyper["value"].append(value)
                    aux_hyperparameter_proba_hyper["probability"].append(chromossome.fitness)
        
    def columns_proba_update(self, chromossome):
        for col in chromossome.cols:
            if col in self.columns_proba["value"]:
                index = self.columns_proba["value"].index(col)
                self.columns_proba["probability"][index] += chromossome.fitness
            else:
                self.columns_proba["value"].append(col)
                self.columns_proba["probability"].append(chromossome.fitness)

    def fit(self, X, y, n_cores, csv_file):
        diversity_values, fitness_values = [], []
        result_dict = dict()
        kf = KFold(n_splits=5, shuffle=False)
        start_time = int(round(time.time() * 1000))
        my_event_loop = asyncio.get_event_loop()
        writing_results_task_obj = None
        
        header = open(csv_file, "w")
        try:
            header.write('start_time,end_time,total_time_ms,diversity,fitness,ensemble_accuracy,ensemble,classifiers_accuracy,ensemble_cols')
            header.write('\n')
        finally:
            header.close()

        selected = []
        not_selected = [x for x in range(0, 2*self.population_size)]
        pop_fitness = []
        all_predictions = np.zeros([2*self.population_size, y.shape[0]])
        y_fit_pred = np.zeros([2*self.population_size, y.shape[0]])
        total_parallel_time = 0

        frequencies = np.unique(y, return_counts=True)[1]
        selection_threshold = max(frequencies)/np.sum(frequencies)
        stop_criteria = 0
        prev_ensemble_accuracy = 0
        best_ensemble_accuracy = 0
        best_ensemble = []
        best_ensemble_cols = []
        best_classifiers_fitness = []
        mutation_guided_before = 1

        for epoch in range(self.max_epochs):
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            time_aux = int(round(now * 1000))

            if stop_criteria == 10:
                break

            ensemble = []
            ensemble_cols = []
            classifiers_fitness = []

            mutation_guided_before = self.generate_offspring(selected, not_selected, pop_fitness, mutation_guided_before)

            parallel_time_aux = int(round(time.time() * 1000))
            backend = 'loky'
            fit_predictions = Parallel(n_jobs=n_cores, backend=backend)(delayed(self.fit_predict_population)(item, kf, X, y) for item in not_selected)
            total_parallel_time = total_parallel_time + (int(round(time.time() * 1000)) - parallel_time_aux)
            
            for i in fit_predictions:
                all_predictions[i[0]] = i[1]
                y_fit_pred[i[0]] = i[2]

            selected, diversity, fitness, pop_fitness = self.diversity_selection(all_predictions, selection_threshold)
            not_selected = np.setdiff1d([x for x in range(0, 2*self.population_size)], selected)
            
            len_X = len(X)
            if (len(selected) < self.population_size):
                diff = self.population_size - len(selected)
                for i in range(0, diff):
                    extra_y_train_pred = np.zeros(len_X)
                    extra_predictions = np.zeros([y.shape[0]])
                    extra_classifier = Chromossome(genotypes_pool=self.algorithms, x_n_cols=self.x_n_cols, col_as_gene=self.col_as_gene, guided_mutation=self.guided_mutation, random_state=self.random_state)
                    extra_classifier.mutate(self.rnd, self.hyperparameter_proba, self.columns_proba)                   
                    for train, val in kf.split(X):
                        if self.col_as_gene == 1:
                            extra_classifier.fit(X[train][:,extra_classifier.cols], y[train])
                            extra_y_train_pred[val] = extra_classifier.predict(X[val][:,extra_classifier.cols])
                        else:
                            extra_classifier.fit(X[train], y[train])
                            extra_y_train_pred[val] = extra_classifier.predict(X[val])
                        extra_predictions[val] = np.equal(extra_y_train_pred[val], y[val])
                    extra_classifier.fitness = extra_predictions.sum()/len(extra_predictions)
                    y_fit_pred[not_selected[i]] = extra_y_train_pred
                    self.population[not_selected[i]] = extra_classifier
                    selected.append(not_selected[i])
                    not_selected = np.delete(not_selected, i)

            ensemble_pred = np.zeros([self.population_size, len_X])
            for i, sel in enumerate(selected):
                chromossome = self.population[sel]
                if self.guided_mutation == 1:
                    #Update aux_hyperparameter_proba_hyper
                    self.hyperparameter_proba_update(chromossome)
                    if self.col_as_gene == 1:
                        #Update aux_columns_proba
                        self.columns_proba_update(chromossome)
                ensemble.append(chromossome.classifier)
                if self.col_as_gene == 1:
                    ensemble_cols.append(chromossome.cols)
                classifiers_fitness.append(chromossome.fitness)
                ensemble_pred[i] = y_fit_pred[sel]
                
            y_train_pred = np.zeros(len_X)
            for i in range(0, len_X):
                pred = {}
                for j, sel in enumerate(selected):
                    if ensemble_pred[j][i] in pred:
                        pred[ensemble_pred[j][i]] += self.population[sel].fitness
                    else:
                        pred[ensemble_pred[j][i]]  = self.population[sel].fitness
                y_train_pred[i] = max(pred.items(), key=operator.itemgetter(1))[0]

            ensemble_accuracy = accuracy_score(y, y_train_pred)
                
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            end_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            total_time = (int(round(now * 1000)) - time_aux)
            
            if (epoch%100 == 0 and epoch != 0):
                if (writing_results_task_obj is not None):
                    my_event_loop.run_until_complete(writing_results_task_obj)
                writing_results_task_obj = my_event_loop.create_task(writing_results_task(result_dict, csv_file))
                result_dict = dict()
            result_dict.update({epoch:{"start_time":start_time,
                                       "end_time":end_time,
                                       "total_time_ms":total_time,
                                       "diversity":diversity,
                                       "fitness":fitness,
                                       "ensemble_accuracy":ensemble_accuracy,
                                       "ensemble":ensemble, 
                                       "classifiers_accuracy":classifiers_fitness,
                                       "ensemble_cols":ensemble_cols
                                       }})                
            if prev_ensemble_accuracy != 0:
                increase_accuracy = ((ensemble_accuracy - prev_ensemble_accuracy)/prev_ensemble_accuracy) * 100.0
                if (increase_accuracy < 0.5):
                    stop_criteria = stop_criteria + 1
                else:
                    stop_criteria = 0
            prev_ensemble_accuracy = ensemble_accuracy
            
            if best_ensemble_accuracy < ensemble_accuracy:
                best_ensemble_accuracy = ensemble_accuracy
                best_ensemble = ensemble
                best_ensemble_cols = ensemble_cols
                best_classifiers_fitness = classifiers_fitness
            
        writing_results_task_obj = my_event_loop.create_task(writing_results_task(result_dict, csv_file))
        my_event_loop.run_until_complete(writing_results_task_obj)
        result_dict = dict()
        print("\n>>>>> Parallel step processing time = %i" % (total_parallel_time))
        return best_ensemble, best_classifiers_fitness, best_ensemble_cols
    
    def fit_ensemble(self, X, y, ensemble, classifiers_fitness, ensemble_cols):
        classifiers_fitness_it = 0
        for index, estimator in enumerate(ensemble):
            if self.col_as_gene == 1:
                self.ensemble.append(Estimator(classifier=estimator, random_state=self.random_state, fitness=classifiers_fitness[classifiers_fitness_it], ensemble_cols=ensemble_cols[index]))
            else:
                self.ensemble.append(Estimator(classifier=estimator, random_state=self.random_state, fitness=classifiers_fitness[classifiers_fitness_it], ensemble_cols=[]))
            classifiers_fitness_it = classifiers_fitness_it + 1
        for classifier in self.ensemble:
            if self.col_as_gene == 1:
                classifier.fit(X[:,classifier.cols], y)
            else:
                classifier.fit(X, y)

    def predict(self, X):
        len_X = len(X)
        predictions = np.zeros((self.population_size, len_X))
        y = np.zeros(len_X)
        for chromossome in range(0, self.population_size):
            if self.col_as_gene == 1:
                predictions[chromossome] = self.ensemble[chromossome].predict(X[:,self.ensemble[chromossome].cols])
            else:
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
    
async def writing_results_task(result_dict, csv_file):
    async with AIOFile(csv_file, 'a') as afp:
        writer = Writer(afp)
        await writer(pd.DataFrame.from_dict(result_dict, orient='index').to_csv(header=False, index=None))
        await afp.fsync()
    
def compare_results(data, target, n_estimators, outputfile, stop_time, n_cores, col_as_gene, guided_mutation):
    accuracy, f1, precision, recall, auc = 0, 0, 0, 0, 0
    total_accuracy, total_f1, total_precision, total_recall, total_auc = [], [], [], [], []
    sum_total_iter_time = []
    fit_total_time = 0
    alg = gen_members(data.shape)
    
    with open(outputfile, "w") as text_file:
        text_file.write('*'*60)
        text_file.write(' PDCE ')
        text_file.write('*'*60)
        text_file.write('\n\nn_estimators = %i' % (n_estimators))
        text_file.write('\nstop_time = %i' % (stop_time))
        kf = KFold(n_splits=5, shuffle=False)
        for i in range(0, 10):
            fold = 0
            print('\n\nIteration = ',i)
            text_file.write("\n\nIteration = %i" % (i))
            for train, val in kf.split(data):
                print('\n\n>>>>>>>>>> Fold = ',fold)
                text_file.write("\n\n>>>>>>>>>> Fold = %i" % (fold))
                fit_time_aux = int(round(time.time() * 1000))
                csv_file = 'pdce_fold_' + str(fold) + '_col_' + str(col_as_gene) + '_mutation_' + str(guided_mutation)  + '_iter_' + str(i) + '_cores_' + str(n_cores) + '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv' 
                ensemble_classifier = DiversityEnsembleClassifier(algorithms=alg,
                                                                  x_n_cols=data[train].shape[1],
                                                                  col_as_gene=col_as_gene,
                                                                  guided_mutation=guided_mutation,
                                                                  population_size=n_estimators, 
                                                                  max_epochs=stop_time,
                                                                  random_state=i*10
                                                                  )
                ensemble, classifiers_fitness, ensemble_cols = ensemble_classifier.fit(data[train], target[train], n_cores, csv_file)
                ensemble_classifier.fit_ensemble(data[train], target[train], ensemble, classifiers_fitness, ensemble_cols)
                fit_total_time = (int(round(time.time() * 1000)) - fit_time_aux)
                text_file.write("\n\nDEC fit done in %i" % (fit_total_time))
                text_file.write(" ms")
                predict_aux = int(round(time.time() * 1000))
                y_pred = ensemble_classifier.predict(data[val])
                predict_total_time = (int(round(time.time() * 1000)) - predict_aux)
                text_file.write("\n\nDEC predict done in %i" % (predict_total_time))
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
                total_iter_time = (int(round(time.time() * 1000)) - fit_time_aux)
                text_file.write("\nIteration done in %i" % (total_iter_time))
                text_file.write(" ms")
                print("\n>>>>> Iteration done in %i" % (total_iter_time))
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
        opts, args = getopt.getopt(argv,"h:i:o:e:s:c:g:m:",["ifile=","ofile=","enumber=","stoptime=","cores=","cgene=","mutation="])
    except getopt.GetoptError:
        print('pdce.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores> -g <col_as_gene> -m <guided_mutation>')
        sys.exit(2)
    if opts == []:
        print('pdce.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores> -g <col_as_gene> -m <guided_mutation>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('pdce.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time> -c <n_cores> -g <col_as_gene> -m <guided_mutation>')
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
        elif opt in ("-g", "--cgene"):
            col_as_gene = int(arg)
        elif opt in ("-m", "--mutation"):
            guided_mutation = int(arg)
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    print('The number of estimators is ', n_estimators)
    print('The number of iterations is ', stop_time)
    print('The number of cores is ', n_cores)

    if col_as_gene == 1:
        print("The columns will be considered as genes by the algorithm")
    elif col_as_gene == 0:
        print("The columns will not be considered as genes by the algorithm")

    if guided_mutation == 1:
        print("The mutation will be guided by the fitness")
    elif guided_mutation == 0:
        print("The mutation will not be guided by the fitness")
    
    if col_as_gene == 1 or col_as_gene == 0:
        if guided_mutation == 1 or guided_mutation == 0:
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            print('\nStart time = ', start_time)
            print('\n')
            
            if inputfile == "iris":
                dataset = datasets.load_iris()
                print('Runing PDCE with guided mutation (cols as gene version)...')
                compare_results(data=dataset.data, 
                                target=dataset.target, 
                                n_estimators=int(n_estimators), 
                                outputfile=outputfile, 
                                stop_time=int(stop_time),
                                n_cores=int(n_cores),
                                col_as_gene=col_as_gene,
                                guided_mutation=guided_mutation
                            )
            elif inputfile == "breast":
                dataset = datasets.load_breast_cancer()
                print('Runing PDCE with guided mutation (cols as gene version)...')
                compare_results(data=dataset.data, 
                                target=dataset.target, 
                                n_estimators=int(n_estimators), 
                                outputfile=outputfile, 
                                stop_time=int(stop_time),
                                n_cores=int(n_cores),
                                col_as_gene=col_as_gene,
                                guided_mutation=guided_mutation
                            )
            elif  inputfile == "wine":
                dataset = datasets.load_wine()
                print('Runing PDCE with guided mutation (cols as gene version)...')
                compare_results(data=dataset.data, 
                                target=dataset.target, 
                                n_estimators=int(n_estimators), 
                                outputfile=outputfile, 
                                stop_time=int(stop_time),
                                n_cores=int(n_cores),
                                col_as_gene=col_as_gene,
                                guided_mutation=guided_mutation
                            )
            else:
                le = LabelEncoder()
                dataset = pd.read_csv(inputfile)
                dataset.iloc[:, -1] = le.fit_transform(dataset.iloc[:, -1])
                print('Runing PDCE with guided mutation (cols as gene version)...')
                compare_results(data=dataset.iloc[:, 0:-1].values, 
                                target=dataset.iloc[:, -1].values, 
                                n_estimators=int(n_estimators), 
                                outputfile=outputfile, 
                                stop_time=int(stop_time),
                                n_cores=int(n_cores),
                                col_as_gene=col_as_gene,
                                guided_mutation=guided_mutation
                            )
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            end_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            print('\nEnd time = ', end_time)
            print('\nIt is finished!')
        else:
            print("ERROR - The parameter guided_mutation must be defined as 0 or 1")
    else:
        print("ERROR - The parameter col_as_gene must be defined as 0 or 1")

if __name__ == "__main__":
    main(sys.argv[1:])