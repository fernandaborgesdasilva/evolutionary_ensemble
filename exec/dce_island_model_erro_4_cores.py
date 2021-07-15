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
from numpy.random import RandomState, SeedSequence
import pandas as pd
import statistics
import operator
import time
import sys, getopt, os
import copy
from collections import defaultdict
from scipy.stats import truncnorm
from joblib import Parallel, delayed
from all_members_ensemble import gen_members
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

class Individual:
    def __init__(self, genotypes_pool, len_X, x_n_cols, rnd=None, random_state=None):
        self.genotypes_pool = genotypes_pool
        self.classifier = None
        self.classifier_algorithm = None
        self.fitness = 0
        self.y_train_pred = np.zeros(len_X)
        self.rnd = rnd
        self.random_state = random_state
        self.cols = []
        self.x_n_cols = x_n_cols

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def mutate(self, hyperparameter_proba, columns_proba, island_id, mutation_guided_before=0, extra_classifier_to_mutate=0, n_positions=None):
        change = self.rnd.randint(0, 3)

        test_file_name = "teste_4_cores_ilha_" + str(island_id) + ".txt"
        test_file = open(test_file_name, "a")
    
        test_file.write('\n')
        test_file.write(' change = ' + str(change))
        test_file.write('\n')

        if self.classifier is None or extra_classifier_to_mutate == 1:
            param = {}
            #defining the classifier
            self.classifier_algorithm = list(self.genotypes_pool.keys())[self.rnd.choice(len(list(self.genotypes_pool.keys())))]
            
            test_file.write('\n')
            test_file.write(' self.classifier_algorithm = ' + str(self.classifier_algorithm))
            test_file.write('\n')
            
            mod, f = self.classifier_algorithm.rsplit('.', 1)
            clf = getattr(__import__(mod, fromlist=[f]), f)()
            #defining the hyperparameters conf
            for hyperparameter, h_range in self.genotypes_pool[self.classifier_algorithm].items():
                if isinstance(h_range[0], str):
                    param[hyperparameter] = h_range[self.rnd.choice(len(h_range))]

                    test_file.write('\n')
                    test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                    test_file.write('\n')

                elif isinstance(h_range[0], float):
                    h_range_ = []
                    h_range_.append(min(h_range))
                    h_range_.append(max(h_range))
                    param[hyperparameter] = self.rnd.uniform(h_range_[0], h_range_[1])

                    test_file.write('\n')
                    test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                    test_file.write('\n')

                else:
                    h_range_ = []
                    h_range_.append(min(h_range))
                    h_range_.append(max(h_range))
                    param[hyperparameter] = self.rnd.randint(h_range_[0], h_range_[1]+1)

                    test_file.write('\n')
                    test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                    test_file.write('\n')

            self.classifier = clf.set_params(**param)
            all_parameters = self.classifier.get_params()
            if 'random_state' in list(all_parameters.keys()):
                self.classifier.set_params(random_state=self.random_state)
            #defining which columns will be used
            n_cols = self.rnd.randint(1, self.x_n_cols + 1)

            test_file.write('\n')
            test_file.write(' n_cols = ' + str(n_cols))
            test_file.write('\n')

            self.cols = self.rnd.choice(self.x_n_cols, n_cols, replace=False)

            test_file.write('\n')
            test_file.write(' self.cols = ' + str(self.cols))
            test_file.write('\n')

        elif change == 0:
            #changing the classifier
            param = {}
            self.classifier_algorithm = list(self.genotypes_pool.keys())[self.rnd.choice(len(list(self.genotypes_pool.keys())))]
            
            test_file.write('\n')
            test_file.write(' self.classifier_algorithm = ' + str(self.classifier_algorithm))
            test_file.write('\n')

            mod, f = self.classifier_algorithm.rsplit('.', 1)
            clf = getattr(__import__(mod, fromlist=[f]), f)()
            #defining the hyperparameters conf
            for hyperparameter, h_range in self.genotypes_pool[self.classifier_algorithm].items():
                if len(h_range) > 1:
                    hyper_values = []
                    hyper_proba = []
                    hyper_proba_ = []
                    if hyperparameter_proba.get(self.classifier_algorithm, 0) != 0:
                        if hyperparameter_proba[self.classifier_algorithm].get(hyperparameter, 0) != 0:
                            hyper_values = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["value"]
                            hyper_proba = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["probability"]
                    for fitness in hyper_proba:
                        if sum(hyper_proba) > 0:
                            hyper_proba_.append(fitness/sum(hyper_proba))
                        else:
                            hyper_proba_.append(1.0/len(hyper_proba))

                    if mutation_guided_before == 0 or len(hyper_values) == 0:
                        #it does a search for new values
                        if isinstance(h_range[0], str):
                            param[hyperparameter] = h_range[self.rnd.choice(len(h_range))]
                            
                            test_file.write('\n')
                            test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                            test_file.write('\n')

                        elif isinstance(h_range[0], float):
                            h_range_ = []
                            h_range_.append(min(h_range))
                            h_range_.append(max(h_range))
                            param[hyperparameter] = self.rnd.uniform(h_range_[0], h_range_[1])

                            test_file.write('\n')
                            test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                            test_file.write('\n')

                        else:
                            h_range_ = []
                            h_range_.append(min(h_range))
                            h_range_.append(max(h_range))
                            param[hyperparameter] = self.rnd.randint(h_range_[0], h_range_[1] + 1)

                            test_file.write('\n')
                            test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                            test_file.write('\n')
                    else:
                        #it chooses between the values already used
                        if isinstance(h_range[0], str):
                            param[hyperparameter] = self.rnd.choice(hyper_values, 1, p=hyper_proba_).item()

                            test_file.write('\n')
                            test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                            test_file.write('\n')

                        elif isinstance(h_range[0], float):
                            mu = float(self.rnd.choice(hyper_values, 1, p=hyper_proba_))

                            test_file.write('\n')
                            test_file.write(' mu = ' + str(mu))
                            test_file.write('\n')

                            sigma = (max(h_range) - min(h_range))/80
                            a,b = (min(h_range)-mu)/sigma, (max(h_range)-mu)/sigma
                            param[hyperparameter] = truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=self.random_state)
                        else:
                            r_proba_val = int(self.rnd.choice(hyper_values, 1, p=hyper_proba_))

                            test_file.write('\n')
                            test_file.write(' r_proba_val = ' + str(r_proba_val))
                            test_file.write('\n')

                            min_noise = max(min(h_range), r_proba_val - 2)
                            max_noise = min(max(h_range), r_proba_val + 2)
                            param[hyperparameter] = self.rnd.choice(list(set(range(min_noise, max_noise + 1)) - set([r_proba_val])), 1).item()

                            test_file.write('\n')
                            test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                            test_file.write('\n')

                elif len(h_range) == 1:
                    param[hyperparameter] = h_range[0]
            self.classifier = clf.set_params(**param)
            all_parameters = self.classifier.get_params()
            if 'random_state' in list(all_parameters.keys()):
                self.classifier.set_params(random_state=self.random_state)
        elif change == 1:
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
                        n_positions = self.rnd.randint(1, len(possible_hypers) + 1)

                        test_file.write('\n')
                        test_file.write(' n_positions = ' + str(n_positions))
                        test_file.write('\n')

                    mutation_positions = self.rnd.choice(range(0, len(possible_hypers)), n_positions)
                    
                    test_file.write('\n')
                    test_file.write(' mutation_positions = ' + str(mutation_positions))
                    test_file.write('\n')

                    for hyper_id in mutation_positions:
                        hyperparameter = possible_hypers[hyper_id]
                        h_range = self.genotypes_pool[self.classifier_algorithm][hyperparameter]
                        hyper_values = []
                        hyper_proba = []
                        hyper_proba_ = []
                        if hyperparameter_proba.get(self.classifier_algorithm, 0) != 0:
                            if hyperparameter_proba[self.classifier_algorithm].get(hyperparameter, 0) != 0:
                                hyper_values = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["value"]
                                hyper_proba = hyperparameter_proba[self.classifier_algorithm][hyperparameter]["probability"]
                        for fitness in hyper_proba:
                            if sum(hyper_proba) > 0:
                                hyper_proba_.append(fitness/sum(hyper_proba))
                            else:
                                hyper_proba_.append(1.0/len(hyper_proba))

                        if mutation_guided_before == 0 or len(hyper_values) == 0:
                            #it does a search for new values
                            if isinstance(h_range[0], str):
                                param[hyperparameter] = h_range[self.rnd.choice(len(h_range))]

                                test_file.write('\n')
                                test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                                test_file.write('\n')

                            elif isinstance(h_range[0], float):
                                h_range_ = []
                                h_range_.append(min(h_range))
                                h_range_.append(max(h_range))
                                param[hyperparameter] = self.rnd.uniform(h_range_[0], h_range_[1])

                                test_file.write('\n')
                                test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                                test_file.write('\n')

                            else:
                                h_range_ = []
                                h_range_.append(min(h_range))
                                h_range_.append(max(h_range))
                                param[hyperparameter] = self.rnd.randint(h_range_[0], h_range_[1] + 1)

                                test_file.write('\n')
                                test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                                test_file.write('\n')

                        else:
                            #it chooses between the values already used
                            if isinstance(h_range[0], str):
                                param[hyperparameter] = self.rnd.choice(hyper_values, 1, p=hyper_proba_).item()

                                test_file.write('\n')
                                test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                                test_file.write('\n')

                            elif isinstance(h_range[0], float):
                                mu = float(self.rnd.choice(hyper_values, 1, p=hyper_proba_))

                                test_file.write('\n')
                                test_file.write(' mu = ' + str(mu))
                                test_file.write('\n')

                                sigma = (max(h_range) - min(h_range))/80
                                a,b = (min(h_range)-mu)/sigma, (max(h_range)-mu)/sigma
                                param[hyperparameter] = truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=self.random_state)
                            else:
                                r_proba_val = int(self.rnd.choice(hyper_values, 1, p=hyper_proba_))

                                test_file.write('\n')
                                test_file.write(' r_proba_val = ' + str(r_proba_val))
                                test_file.write('\n')

                                min_noise = max(min(h_range), r_proba_val - 2)
                                max_noise = min(max(h_range), r_proba_val + 2)
                                param[hyperparameter] = self.rnd.choice(list(set(range(min_noise, max_noise + 1)) - set([r_proba_val])), 1).item()

                                test_file.write('\n')
                                test_file.write(' param[hyperparameter] = ' + str(param[hyperparameter]))
                                test_file.write('\n')

                self.classifier = clf.set_params(**param)
                all_parameters = self.classifier.get_params()
                if 'random_state' in list(all_parameters.keys()):
                    self.classifier.set_params(random_state=self.random_state)
        elif change == 2:
            #changing the columns
            cols_proba_ = []
            cols_values = columns_proba["value"]
            cols_proba = columns_proba["probability"]
            for fitness in cols_proba:
                if sum(cols_proba) > 0:
                    cols_proba_.append(fitness/sum(cols_proba))
                else:
                    cols_proba_.append(1.0/len(cols_proba))
            n_cols = self.rnd.randint(1, self.x_n_cols + 1)

            test_file.write('\n')
            test_file.write(' n_cols = ' + str(n_cols))
            test_file.write('\n')

            if mutation_guided_before == 0 or n_cols > len(cols_values):
                self.cols = self.rnd.choice(self.x_n_cols, n_cols, replace=False)

                test_file.write('\n')
                test_file.write(' self.cols = ' + str(self.cols))
                test_file.write('\n')

            else:
                self.cols = self.rnd.choice(cols_values, n_cols, p=cols_proba_, replace=False)

                test_file.write('\n')
                test_file.write(' self.cols = ' + str(self.cols))
                test_file.write('\n')

        test_file.close()
            
class Estimator:
    def __init__(self, classifier=None, random_state=None, fitness=0):
        self.classifier = classifier
        self.fitness = fitness

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

class DiversityEnsembleClassifier:
    def __init__(self, individuals, len_X, x_n_cols, population_size, num_generations, num_islands, migration_interval, migration_size, random_state):
        self.population_size = population_size
        self.num_generations = num_generations
        self.individuals = individuals
        self.len_X = len_X
        self.x_n_cols = x_n_cols
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.random_state = random_state
        self.rnd = RandomState(self.random_state)
        self.ensemble = []
        self.hyperparameter_proba = [defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for _ in range(self.num_islands)]
        self.columns_proba = [{"value":[],"probability":[]} for _ in range(self.num_islands)]
        #self.islands_results_before_migrate = [ [None] * 6 ] * self.num_islands
        self.islands_results_before_migrate = [{"best_ensemble":None,
                                                "best_classifiers_fitness":None,
                                                "best_ensemble_accuracy":None,
                                                "classifiers_fitness":None,
                                                "selected":None,
                                                "mutation_guided_before":None,
                                                "aux_islands_pop":None,
                                                "aux_hyperparameter_proba":None,
                                                "aux_columns_proba":None
                                               } for _ in range(self.num_islands)]
        self.islands_migration_info = [{"idx_individuals_to_migrate":[],"idx_individuals_to_delete":[]} for _ in range(self.num_islands)]
        #self.islands_pop_rnd = [None] * self.num_islands
        self.islands_pop_rnd = []
        self.islands_pop = [ [None] * 2 * self.population_size ] * self.num_islands
        for island_id in range(0, self.num_islands):
            #self.islands_pop_rnd[island_id] = RandomState((island_id + 1)*(self.random_state + 1))
            self.islands_pop_rnd.append(RandomState((island_id + 1)*(self.random_state + 1)))
            for i in range(0, self.population_size):
                self.islands_pop[island_id][i] = Individual(genotypes_pool=self.individuals, len_X=self.len_X, x_n_cols=self.x_n_cols, rnd=self.islands_pop_rnd[island_id], random_state=self.random_state)
            for i in range(0, self.population_size):

                test_file_name = "teste_4_cores_ilha_" + str(island_id) + ".txt"
                test_file = open(test_file_name, "a")
                try:
                    test_file.write('\n\n\n')
                    test_file.write(' MUTACAO NOVOS INDIVIDUOS ')
                    test_file.write('\n')
                finally:
                    test_file.close()

                self.islands_pop[island_id][i].mutate(self.hyperparameter_proba[island_id], self.columns_proba[island_id], island_id)

    def generate_offspring(self, island_id, parents, children, mutation_guided_before, aux_islands_pop, aux_hyperparameter_proba, aux_columns_proba):
        children_aux = children
        if not parents:
            parents = [x for x in range(0, self.population_size)]
            children = [x for x in range(self.population_size, 2*self.population_size)]
        for i in range(0, self.population_size):
            new_individual = copy.deepcopy(aux_islands_pop[island_id][parents[i]])
            new_individual.mutate(aux_hyperparameter_proba[island_id], aux_columns_proba[island_id], island_id, mutation_guided_before)
            aux_islands_pop[island_id][children[i]] = new_individual
        if mutation_guided_before == 0:
            return 1
        else:
            return 0

    def fit_predict_population(self, island_id, not_fitted, predictions, kfolds, X, y, aux_islands_pop):
        for i in not_fitted:
            individual = aux_islands_pop[island_id][i]
            for train, val in kfolds.split(X):
                individual.fit(X[train][:,individual.cols], y[train])
                individual.y_train_pred[val] = individual.predict(X[val][:,individual.cols])
                predictions[i][val] = np.equal(individual.y_train_pred[val], y[val])
        return predictions

    def diversity_selection(self, island_id, predictions, selection_threshold, aux_islands_pop):
        distances = np.zeros(2*self.population_size)
        pop_fitness = predictions.sum(axis=1)/predictions.shape[1]
        target_individual = np.argmax(pop_fitness)
        selected = [target_individual]
        aux_islands_pop[island_id][target_individual].fitness = pop_fitness[target_individual]
        diversity  = np.zeros(2*self.population_size)
        mean_fitness = pop_fitness[target_individual]
        distances[pop_fitness < selection_threshold] = float('-inf')
        for i in range(0, self.population_size-1):
            distances[target_individual] = float('-inf')
            d_i = np.logical_xor(predictions, predictions[target_individual]).sum(axis=1)
            distances += d_i
            target_individual = np.argmax(distances)
            if distances[target_individual] == float('-inf'):
                break
            diversity += d_i/predictions.shape[1]
            mean_fitness += pop_fitness[target_individual]
            selected.append(target_individual)
            aux_islands_pop[island_id][target_individual].fitness = pop_fitness[target_individual]
        return selected, (diversity[selected]/self.population_size).mean(), mean_fitness/(self.population_size)
    
    def hyperparameter_proba_update(self, individual, island_id, aux_hyperparameter_proba):
        for hyper in individual.classifier.get_params():
            if hyper in self.individuals[individual.classifier_algorithm].keys():
                if isinstance(individual.classifier.get_params()[hyper], float):
                    value = round(individual.classifier.get_params()[hyper], 3)
                else:
                    value = individual.classifier.get_params()[hyper]
                    assert isinstance(value, int) or isinstance(value, str), 'hyperparameter value of wrong type!'
                aux_hyperparameter_proba_hyper = aux_hyperparameter_proba[island_id][individual.classifier_algorithm][hyper]
                if aux_hyperparameter_proba[island_id][individual.classifier_algorithm].get(hyper, 0) != 0:
                #this hyperparameter already exists in aux_hyperparameter_proba
                    if value in aux_hyperparameter_proba_hyper['value']:
                    #this hyperparameter value already exists in aux_hyperparameter_proba
                        index = aux_hyperparameter_proba_hyper["value"].index(value)
                        aux_hyperparameter_proba_hyper["probability"][index] += individual.fitness
                    else:
                    #adding this hyperparameter value to aux_hyperparameter_proba
                        if len(aux_hyperparameter_proba_hyper["value"]) < 20:
                            aux_hyperparameter_proba_hyper["value"].append(value)
                            aux_hyperparameter_proba_hyper["probability"].append(individual.fitness)
                        else:
                            min_proba = min(aux_hyperparameter_proba_hyper["probability"])
                            min_proba_index = aux_hyperparameter_proba_hyper["probability"].index(min_proba)
                            del aux_hyperparameter_proba_hyper["probability"][min_proba_index]
                            del aux_hyperparameter_proba_hyper["value"][min_proba_index]
                            aux_hyperparameter_proba_hyper["value"].append(value)
                            aux_hyperparameter_proba_hyper["probability"].append(individual.fitness)
                else:
                #adding the hyperparameter to aux_hyperparameter_proba
                    aux_hyperparameter_proba_hyper["value"].append(value)
                    aux_hyperparameter_proba_hyper["probability"].append(individual.fitness)
        return aux_hyperparameter_proba
        
    def columns_proba_update(self, individual, island_id, aux_columns_proba):
        for col in individual.cols:
            if col in aux_columns_proba[island_id]["value"]:
                index = aux_columns_proba[island_id]["value"].index(col)
                aux_columns_proba[island_id]["probability"][index] += individual.fitness
            else:
                aux_columns_proba[island_id]["value"].append(col)
                aux_columns_proba[island_id]["probability"].append(individual.fitness)
        return aux_columns_proba

    def fit(self, X, y, csv_file, island_id):
        result_dict = dict()
        kf = KFold(n_splits=5, random_state=self.random_state)
        start_time = int(round(time.time() * 1000))
        writing_results_task_obj = None
        my_event_loop = asyncio.get_event_loop()
        
        header = open(csv_file, "w")
        try:
            header.write('start_time,end_time,total_time_ms,diversity,fitness,ensemble_accuracy,ensemble,classifiers_accuracy,ensemble_cols')
            header.write('\n')
        finally:
            header.close()
        
        predictions = np.zeros([2*self.population_size, y.shape[0]])
        frequencies = np.unique(y, return_counts=True)[1]
        selection_threshold = max(frequencies)/np.sum(frequencies)
        aux_islands_pop = copy.deepcopy(self.islands_pop)
        aux_hyperparameter_proba = copy.deepcopy(self.hyperparameter_proba)
        aux_columns_proba = copy.deepcopy(self.columns_proba)

        if self.islands_results_before_migrate[island_id]["best_ensemble"] is None:
            selected = []
            not_selected = [x for x in range(0, 2*self.population_size)]
            best_ensemble_accuracy = 0
            best_ensemble = []
            best_classifiers_fitness = []
            mutation_guided_before = 1
        else:
            best_ensemble = copy.deepcopy(self.islands_results_before_migrate[island_id]["best_ensemble"])
            best_classifiers_fitness = copy.deepcopy(self.islands_results_before_migrate[island_id]["best_classifiers_fitness"])
            best_ensemble_accuracy = copy.deepcopy(self.islands_results_before_migrate[island_id]["best_ensemble_accuracy"])
            classifiers_fitness = copy.deepcopy(self.islands_results_before_migrate[island_id]["classifiers_fitness"])
            selected = copy.deepcopy(self.islands_results_before_migrate[island_id]["selected"])
            mutation_guided_before = copy.deepcopy(self.islands_results_before_migrate[island_id]["mutation_guided_before"])
            not_selected = np.setdiff1d([x for x in range(0, 2*self.population_size)], selected)
        
        for epoch in range(self.migration_interval):
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            time_aux = int(round(now * 1000))
            ensemble = []
            ensemble_cols = []
            classifiers_fitness = []

            test_file_name = "teste_4_cores_ilha_" + str(island_id) + ".txt"
            test_file = open(test_file_name, "a")
            try:
                test_file.write('\n\n\n')
                test_file.write(' MUTACAO NORMAL ')
                test_file.write('\n')
            finally:
                test_file.close()

            #This step generates new individuals by mutating the selected ones in the generation before
            mutation_guided_before = self.generate_offspring(island_id, selected, not_selected, mutation_guided_before, aux_islands_pop, aux_hyperparameter_proba, aux_columns_proba)
            predictions = self.fit_predict_population(island_id, not_selected, predictions, kf, X, y, aux_islands_pop)
            #This selection step of individuals is guided by diversity criteria
            selected, diversity, fitness = self.diversity_selection(island_id, predictions, selection_threshold, aux_islands_pop)
            not_selected = np.setdiff1d([x for x in range(0, 2*self.population_size)], selected)
            len_X = len(X)
            if (len(selected) < self.population_size):
                diff = self.population_size - len(selected)
                for i in range(0, diff):
                    extra_predictions = np.zeros([y.shape[0]])
                    extra_classifier = Individual(genotypes_pool=self.individuals, len_X=self.len_X, x_n_cols=self.x_n_cols, rnd=self.islands_pop_rnd[island_id], random_state=self.random_state)
                    #extra_classifier = copy.deepcopy(aux_islands_pop[island_id][len(selected)-1])
                    extra_classifier_to_mutate = 1

                    test_file = open(test_file_name, "a")
                    try:
                        test_file.write('\n\n\n')
                        test_file.write(' MUTACAO INDIVIDUOS EXTRAS ')
                        test_file.write('\n')
                    finally:
                        test_file.close()

                    extra_classifier.mutate(aux_hyperparameter_proba[island_id], aux_columns_proba[island_id], island_id, extra_classifier_to_mutate)
                    for train, val in kf.split(X):
                        extra_classifier.fit(X[train][:,extra_classifier.cols], y[train])
                        extra_classifier.y_train_pred[val] = extra_classifier.predict(X[val][:,extra_classifier.cols])
                        extra_predictions[val] = np.equal(extra_classifier.y_train_pred[val], y[val])
                    extra_classifier.fitness = extra_predictions.sum()/len(extra_predictions)
                    aux_islands_pop[island_id][not_selected[i]] = extra_classifier
                    selected.append(not_selected[i])
                    not_selected = np.delete(not_selected, i)
            #The weight of the probability of a value of hyperparameter being chosen is increased in this step
            ensemble_pred = np.zeros([self.population_size, len_X])
            for i, sel in enumerate(selected):
                individual = aux_islands_pop[island_id][sel]
                #Update aux_hyperparameter_proba_hyper
                aux_hyperparameter_proba = self.hyperparameter_proba_update(individual, island_id, aux_hyperparameter_proba)
                #Update aux_columns_proba
                aux_columns_proba = self.columns_proba_update(individual, island_id, aux_columns_proba)
                ensemble.append(individual.classifier)
                ensemble_cols.append(individual.cols)
                classifiers_fitness.append(individual.fitness)
                ensemble_pred[i] = individual.y_train_pred
            #Here the weighted voting is used to combine the classifiers' outputs by considering the strength of the classifiers prior to voting
            y_train_pred = np.zeros(len_X) 
            for i in range(0, len_X):
                pred = {}
                for j, sel in enumerate(selected):
                    if ensemble_pred[j][i] in pred:
                        pred[ensemble_pred[j][i]] += aux_islands_pop[island_id][sel].fitness
                    else:
                        pred[ensemble_pred[j][i]] = aux_islands_pop[island_id][sel].fitness
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
            if best_ensemble_accuracy < ensemble_accuracy:
                best_ensemble_accuracy = ensemble_accuracy
                best_ensemble = ensemble
                best_classifiers_fitness = classifiers_fitness

        writing_results_task_obj = my_event_loop.create_task(writing_results_task(result_dict, csv_file))
        my_event_loop.run_until_complete(writing_results_task_obj)
        result_dict = dict()
        dict_to_return = dict()
        dict_to_return.update({"best_ensemble":best_ensemble,
                               "best_classifiers_fitness":best_classifiers_fitness,
                               "best_ensemble_accuracy":best_ensemble_accuracy,
                               "classifiers_fitness":classifiers_fitness,
                               "selected":selected,
                               "mutation_guided_before":mutation_guided_before,
                               "aux_islands_pop":aux_islands_pop,
                               "aux_hyperparameter_proba":aux_hyperparameter_proba,
                               "aux_columns_proba":aux_columns_proba
                              })
        return dict_to_return

    def fit_islands(self, X, y, csv_file_name_beg, n_cores):
        best_ensemble_accuracy = 0
        prev_ensemble_accuracy = 0
        csv_file_name_end = '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
        backend = 'loky'
        for iteration in range(0, round(self.num_generations/self.migration_interval)):
            csv_file_name_end = '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
            fit_results = Parallel(n_jobs=n_cores, backend=backend)(delayed(self.fit)(X, y, csv_file_name_beg + '_island_' + str(isl) + csv_file_name_end, isl) for isl in range(0, self.num_islands))

            for ild in range(0, self.num_islands):
                test_file_name = "teste_4_cores_ilha_" + str(ild) + ".txt"
                test_file = open(test_file_name, "a")
                try:
                    test_file.write('\n\n\n')
                    test_file.write(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
                    test_file.write('\n')
                    test_file.write(' NOVA MIGRACAO ')
                    test_file.write('\n')
                    test_file.write(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
                    test_file.write('\n')
                finally:
                    test_file.close()

            for island_id, island in enumerate(fit_results):
                self.islands_pop[island_id] = copy.deepcopy(island["aux_islands_pop"][island_id])
                self.hyperparameter_proba[island_id] = copy.deepcopy(island["aux_hyperparameter_proba"][island_id])
                self.columns_proba[island_id] = copy.deepcopy(island["aux_columns_proba"][island_id])
                self.islands_results_before_migrate[island_id]["best_ensemble"] = copy.deepcopy(island["best_ensemble"])
                self.islands_results_before_migrate[island_id]["best_classifiers_fitness"] = copy.deepcopy(island["best_classifiers_fitness"])
                self.islands_results_before_migrate[island_id]["best_ensemble_accuracy"] = copy.deepcopy(island["best_ensemble_accuracy"])
                self.islands_results_before_migrate[island_id]["classifiers_fitness"] = copy.deepcopy(island["classifiers_fitness"])
                self.islands_results_before_migrate[island_id]["selected"] = copy.deepcopy(island["selected"])
                self.islands_results_before_migrate[island_id]["mutation_guided_before"] = copy.deepcopy(island["mutation_guided_before"])
                last_pop_classifiers_fitness = np.array(island["classifiers_fitness"]).argsort()
                self.islands_migration_info[island_id]["idx_individuals_to_migrate"] = last_pop_classifiers_fitness[-self.migration_size:]
                self.islands_migration_info[island_id]["idx_individuals_to_delete"] = last_pop_classifiers_fitness[:self.migration_size]
                
            ring_islands = self.rnd.choice(self.num_islands, self.num_islands, replace=False)
            for island_id, island in enumerate(fit_results):
                right_neighbor_index = list(ring_islands).index(island_id) + 1
                if right_neighbor_index >= len(ring_islands):
                    right_neighbor_index = 0
                neighbor_island = ring_islands[right_neighbor_index]
                last_best_ensemble = self.islands_results_before_migrate[island_id]["best_ensemble"] 
                last_best_classifiers_fitness = self.islands_results_before_migrate[island_id]["best_classifiers_fitness"]
                last_best_ensemble_accuracy = self.islands_results_before_migrate[island_id]["best_ensemble_accuracy"]
                idx_individuals_to_migrate = self.islands_migration_info[island_id]["idx_individuals_to_migrate"]
                idx_individuals_to_delete = self.islands_migration_info[neighbor_island]["idx_individuals_to_delete"]

                for individual in range(0,self.migration_size):
                    self.islands_pop[neighbor_island][idx_individuals_to_delete[individual]] = self.islands_pop[island_id][idx_individuals_to_migrate[individual]]
                if best_ensemble_accuracy < last_best_ensemble_accuracy:
                    best_ensemble_accuracy = last_best_ensemble_accuracy
                    best_ensemble = last_best_ensemble
                    best_classifiers_fitness = last_best_classifiers_fitness
            if prev_ensemble_accuracy != 0:
                increase_accuracy = ((best_ensemble_accuracy - prev_ensemble_accuracy)/prev_ensemble_accuracy) * 100.0
                if (increase_accuracy < 0.5):
                    break
            prev_ensemble_accuracy = best_ensemble_accuracy
    
        return best_ensemble, best_classifiers_fitness

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
        for individual in range(0, self.population_size):
            predictions[individual] = self.ensemble[individual].predict(X)
        #Here the weighted voting is used to combine the classifiers' outputs by considering the strength of the classifiers prior to voting
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

def compare_results(data, target, ensemble_size, outputfile, num_generations, num_islands, migration_interval, migration_size, n_cores):
    accuracy, f1, precision, recall, auc = 0, 0, 0, 0, 0
    total_accuracy, total_f1, total_precision, total_recall, total_auc = [], [], [], [], []
    fit_total_time = 0
    individuals = gen_members(data.shape)
    sum_total_iter_time = []
    
    with open(outputfile, "w") as text_file:
        text_file.write('*'*60)
        text_file.write(' DCE - Island Model ')
        text_file.write('*'*60)
        text_file.write('\n\nensemble_size = %i' % (ensemble_size))
        text_file.write('\nnum_generations = %i' % (num_generations))
        text_file.write('\nnum_islands = %i' % (num_islands))
        text_file.write('\nmigration_interval = %i' % (migration_interval))
        text_file.write('\nmigration_sizel = %i' % (migration_size))
        fold = 0
        kf = KFold(n_splits=5, random_state=42)
        for train, val in kf.split(data):
            print('\n\n>>>>>>>>>> Fold = ',fold)
            text_file.write("\n\n>>>>>>>>>> Fold = %i" % (fold))
            #for i in range(0, 10):
            for i in range(0, 1):
                fit_time_aux = int(round(time.time() * 1000))
                #csv_file = 'dce_island_fold_' + str(fold) + '_iter_' + str(i) + '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
                csv_file_name_beg = 'dce_island_fold_' + str(fold) + '_iter_' + str(i)
                print('\n\nIteration = ',i)
                text_file.write("\n\nIteration = %i" % (i))
                ensemble_classifier = DiversityEnsembleClassifier(individuals=individuals,
                                                                  len_X = len(train),
                                                                  x_n_cols = data[train].shape[1],
                                                                  population_size=ensemble_size, 
                                                                  num_generations=num_generations,
                                                                  num_islands = num_islands,
                                                                  migration_interval = migration_interval,
                                                                  migration_size = migration_size,
                                                                  random_state=i*10)
                #ensemble, classifiers_fitness = ensemble_classifier.fit(data[train], target[train], csv_file_name_beg)
                ensemble, classifiers_fitness = ensemble_classifier.fit_islands(data[train], target[train], csv_file_name_beg, n_cores)
                ensemble_classifier.fit_ensemble(data[train], target[train], ensemble, classifiers_fitness)
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
    ensemble_size = ''
    num_generations = ''
    num_islands = ''
    migration_interval = ''
    migration_size = ''
    n_cores = ''
    try:
        opts, args = getopt.getopt(argv,"h:i:o:e:g:n:m:s:c:",["ifile=",
                                                              "ofile=",
                                                              "esize=",
                                                              "ngenerations=",
                                                              "nislands=",
                                                              "minterval=",
                                                              "msize=",
                                                              "ncores="])
    except getopt.GetoptError:
        print('dce_island_model.py -i <inputfile> -o <outputfile> -e <ensemble_size> -g <num_generations> -n <num_islands> -m <migration_interval> -s <migration_size> -c <n_cores>')
        sys.exit(2)
    if opts == []:
        print('dce_island_model.py -i <inputfile> -o <outputfile> -e <ensemble_size> -g <num_generations> -n <num_islands> -m <migration_interval> -s <migration_size> -c <n_cores>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('dce_island_model.py -i <inputfile> -o <outputfile> -e <ensemble_size> -g <num_generations> -n <num_islands> -m <migration_interval> -s <migration_size> -c <n_cores>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-e", "--esize"):
            ensemble_size = arg
        elif opt in ("-g", "--ngenerations"):
            num_generations = arg
        elif opt in ("-n", "--nislands"):
            num_islands = arg
        elif opt in ("-m", "--minterval"):
            migration_interval = arg
        elif opt in ("-s", "--msize"):
            migration_size = arg
        elif opt in ("-c", "--ncores"):
            n_cores = arg
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    print('The ensemble size is ', ensemble_size)
    print('The number of generations is ', num_generations)
    print('The number of islands is ', num_islands)
    print('The migration interval is ', migration_interval)
    print('The migration size is ', migration_size)
    print('The number of cores is ', n_cores)
    
    now = time.time()
    struct_now = time.localtime(now)
    mlsec = repr(now).split('.')[1][:3]
    start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
    print('\nStart time = ', start_time)
    print('\n')
    
    if inputfile == "iris":
        dataset = datasets.load_iris()
        print('Runing DCE - Island Model...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        ensemble_size=int(ensemble_size), 
                        outputfile=outputfile, 
                        num_generations=int(num_generations),
                        num_islands=int(num_islands),
                        migration_interval=int(migration_interval),
                        migration_size=int(migration_size),
                        n_cores=int(n_cores)
                        )
    elif inputfile == "breast":
        dataset = datasets.load_breast_cancer()
        print('Runing DCE - Island Model...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        ensemble_size=int(ensemble_size), 
                        outputfile=outputfile, 
                        num_generations=int(num_generations),
                        num_islands=int(num_islands),
                        migration_interval=int(migration_interval),
                        migration_size=int(migration_size),
                        n_cores=int(n_cores)
                        )
    elif  inputfile == "wine":
        dataset = datasets.load_wine()
        print('Runing DCE - Island Model...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        ensemble_size=int(ensemble_size), 
                        outputfile=outputfile, 
                        num_generations=int(num_generations),
                        num_islands=int(num_islands),
                        migration_interval=int(migration_interval),
                        migration_size=int(migration_size),
                        n_cores=int(n_cores)
                        )
    else:
        le = LabelEncoder()
        dataset = pd.read_csv(inputfile)
        dataset.iloc[:, -1] = le.fit_transform(dataset.iloc[:, -1])
        print('Runing DCE - Island Model...')
        compare_results(data=dataset.iloc[:, 0:-1].values, 
                        target=dataset.iloc[:, -1].values, 
                        ensemble_size=int(ensemble_size), 
                        outputfile=outputfile, 
                        num_generations=int(num_generations),
                        num_islands=int(num_islands),
                        migration_interval=int(migration_interval),
                        migration_size=int(migration_size),
                        n_cores=int(n_cores)
                        )
    now = time.time()
    struct_now = time.localtime(now)
    mlsec = repr(now).split('.')[1][:3]
    end_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
    print('\nEnd time = ', end_time)
    print('\nIt is finished!')

if __name__ == "__main__":
    main(sys.argv[1:])