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
import statistics
from numpy.random import RandomState, SeedSequence
import operator
import time
import sys, getopt
import copy
from all_members_ensemble import gen_members
import asyncio
from aiofile import AIOFile, Writer
import nest_asyncio
nest_asyncio.apply()

import warnings
import itertools
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Chromossome:
    def __init__(self, genotypes_pool, len_X, random_id, random_state=None):
        self.genotypes_pool = genotypes_pool
        self.classifier = None
        self.classifier_algorithm = None
        self.fitness = 0
        self.y_train_pred = np.zeros(len_X)
        self.random_state = random_state
        self.random_id = self.random_state + random_id
        self.rnd = RandomState(self.random_id)
        self.mutate()

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)
        
    def set_random_id(self, random_id):
        self.random_id = self.random_state + random_id
        self.rnd = RandomState(self.random_id)

    def predict(self, X):
        return self.classifier.predict(X)

    def mutate(self, n_positions=None):
         
        change_classifier = self.rnd.randint(0, len(self.genotypes_pool))
        
        #APAGAR
        if(self.random_state == 0):
            print("\n\n\nDEBUG change_classifier >>>>>> ", change_classifier)


        if self.classifier is None or change_classifier != 0:
            param = {}
            self.classifier_algorithm = list(self.genotypes_pool.keys())[self.rnd.choice(len(list(self.genotypes_pool.keys())))]
            
            #APAGAR
            if(self.random_state == 0):
                print("\nDEBUG self.classifier_algorithm >>>>>> ", self.classifier_algorithm)

            mod, f = self.classifier_algorithm.rsplit('.', 1)
            clf = getattr(__import__(mod, fromlist=[f]), f)()            
        else:
            param = self.classifier.get_params()
            clf = self.classifier

        if not n_positions or n_positions>len(self.genotypes_pool[self.classifier_algorithm]):
            n_positions = len(self.genotypes_pool[self.classifier_algorithm])

        mutation_positions = self.rnd.choice(range(0, len(self.genotypes_pool[self.classifier_algorithm])), n_positions)
        
        #APAGAR
        if(self.random_state == 0):
            print("\nDEBUG mutation_positions >>>>>> ", mutation_positions)
        
        
        i=0
        for hyperparameter, h_range in self.genotypes_pool[self.classifier_algorithm].items():
            if i in mutation_positions or self.classifier_algorithm != self.classifier.__class__:
                if isinstance(h_range[0], str):
                    param[hyperparameter] = h_range[self.rnd.choice(len(h_range))]

                    #APAGAR
                    if(self.random_state == 0):
                        print("\nDEBUG param[hyperparameter] >>>>>> ", param[hyperparameter])


                elif isinstance(h_range[0], float):
                    h_range_ = []
                    h_range_.append(min(h_range))
                    h_range_.append(max(h_range))
                    param[hyperparameter] = self.rnd.uniform(h_range_[0], h_range_[1]+1)

                    #APAGAR
                    if(self.random_state == 0):
                        print("\nDEBUG param[hyperparameter] >>>>>> ", param[hyperparameter])

                else:
                    h_range_ = []
                    h_range_.append(min(h_range))
                    h_range_.append(max(h_range))
                    param[hyperparameter] = self.rnd.randint(h_range_[0], h_range_[1]+1)

                    #APAGAR
                    if(self.random_state == 0):
                        print("\nDEBUG param[hyperparameter] >>>>>> ", param[hyperparameter])
            i+= 1
            
        print("\nDEBUG >>>>>> ", param)    
            
        self.classifier = clf.set_params(**param)

        all_parameters = self.classifier.get_params()

        if 'random_state' in list(all_parameters.keys()):
            self.classifier.set_params(random_state=self.random_state)
            
class Estimator:
    def __init__(self, classifier=None, random_state=None, fitness=0):
        self.classifier = classifier
        #self.random_state = random_state
        self.fitness = fitness

    def fit(self, X, y):
        is_fitted = True
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

class DiversityEnsembleClassifier:
    def __init__(self, algorithms, len_X, population_size = 100, max_epochs = 100, random_state=None):
        self.population_size = population_size
        self.max_epochs = max_epochs
        self.population = []
        self.algorithms = algorithms
        self.random_state = random_state
        self.ensemble = []
        print("\n\n!!!!!!!!!!!! DEBUG DiversityEnsembleClassifier")
        for i in range(0, population_size):
            self.population.append(Chromossome(genotypes_pool=algorithms, len_X=len_X, random_id = i, random_state=self.random_state))

    def generate_offspring(self, parents, children, pop_fitness, epoch):
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
        
        random_id = self.population_size + epoch + 1
        for i in range(0, self.population_size):
            new_chromossome = copy.deepcopy(self.population[parents[i]])
            random_id = random_id + i
            new_chromossome.set_random_id(random_id)
            new_chromossome.mutate()
            try:
                self.population[children[i]] = new_chromossome
            except:
                self.population.append(new_chromossome)

    def fit_predict_population(self, not_fitted, predictions, kfolds, X, y):
        for i in not_fitted:
            chromossome = self.population[i]
            for train, val in kfolds.split(X):
                chromossome.fit(X[train], y[train])
                chromossome.y_train_pred[val] = chromossome.predict(X[val])
                predictions[i][val] = np.equal(chromossome.y_train_pred[val], y[val])
        return predictions

    def diversity_selection(self, predictions, selection_threshold):
        #print(-1)
        distances = np.zeros(2*self.population_size)
        pop_fitness = predictions.sum(axis=1)/predictions.shape[1]
        print("\n\n&&&&&&&&&&&&&&&& DEBUG pop_fitness = ",pop_fitness)
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
    
    def fit(self, X, y, csv_file):
        #diversity_values, fitness_values = [], []
        result_dict = dict()
        ##print('Starting genetic algorithm...')
        kf = KFold(n_splits=5, random_state=self.random_state)
        start_time = int(round(time.time() * 1000))
        writing_results_task_obj = None
        my_event_loop = asyncio.get_event_loop()
        
        header = open(csv_file, "w")
        try:
            header.write('start_time,end_time,total_time_ms,diversity,fitness,ensemble_accuracy,ensemble,classifiers_accuracy')
            header.write('\n')
        finally:
            header.close()

        selected, not_selected, pop_fitness = [], [], []
        predictions = np.zeros([2*self.population_size, y.shape[0]])

        frequencies = np.unique(y, return_counts=True)[1]
        selection_threshold = max(frequencies)/np.sum(frequencies)
        stop_criteria = 0
        prev_ensemble_accuracy = 0
        best_ensemble_accuracy = 0
        best_ensemble = []
        best_classifiers_fitness = []
        
        for epoch in range(self.max_epochs):
            
            now = time.time()
            struct_now = time.localtime(now)
            mlsec = repr(now).split('.')[1][:3]
            start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
            time_aux = int(round(now * 1000))

            if stop_criteria == 10:
                break
            
            ensemble = []
            classifiers_fitness = []
            
            not_selected = np.setdiff1d([x for x in range(0, 2*self.population_size)], selected)
            print("\n\n!!!!!!!!!!!! DEBUG epoch = ", epoch)
            self.generate_offspring(selected, not_selected, pop_fitness, epoch)
            predictions = self.fit_predict_population(not_selected, predictions, kf, X, y)

            #print('Applying diversity selection...', end='')
            selected, diversity, fitness, pop_fitness = self.diversity_selection(predictions, selection_threshold)
            
            #diversity_values.append(diversity)
            #fitness_values.append(fitness)
            len_X = len(X)
            ensemble_pred = np.zeros([self.population_size, len_X])
            y_train_pred = np.zeros(len_X)
            for i, sel in enumerate(selected):
                chromossome = self.population[sel]
                ensemble.append(chromossome.classifier)
                classifiers_fitness.append(chromossome.fitness)
                ensemble_pred[i] = chromossome.y_train_pred
                
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
                                       "classifiers_accuracy":classifiers_fitness}})
            if prev_ensemble_accuracy != 0:
                increase_accuracy = ((ensemble_accuracy - prev_ensemble_accuracy)/prev_ensemble_accuracy) * 100.0
                if (increase_accuracy < 1.0):
                    stop_criteria = stop_criteria + 1
                else:
                    stop_criteria = 0
            prev_ensemble_accuracy = ensemble_accuracy
            
            print("\n\n!!!!!!!!!!!! DEBUG ensemble_accuracy = ", ensemble_accuracy)
            print("!!!!!!!!!!!! DEBUG ensemble = ", ensemble)

            if best_ensemble_accuracy < ensemble_accuracy:
                best_ensemble_accuracy = ensemble_accuracy
                best_ensemble = ensemble
                best_classifiers_fitness = classifiers_fitness

        writing_results_task_obj = my_event_loop.create_task(writing_results_task(result_dict, csv_file))
        my_event_loop.run_until_complete(writing_results_task_obj)
        result_dict = dict()
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

    
async def writing_results_task(result_dict, csv_file):
    async with AIOFile(csv_file, 'a') as afp:
        writer = Writer(afp)
        await writer(pd.DataFrame.from_dict(result_dict, orient='index').to_csv(header=False, index=None))
        await afp.fsync()  
        
def compare_results(data, target, n_estimators, outputfile, stop_time):
    accuracy, f1, precision, recall, auc = 0, 0, 0, 0, 0
    total_accuracy, total_f1, total_precision, total_recall, total_auc = [], [], [], [], []
    fit_total_time = 0
    alg = gen_members(data.shape)
    sum_total_iter_time = []
    
    with open(outputfile, "w") as text_file:
        text_file.write('*'*60)
        text_file.write(' Diversity-based Ensemble Classifier ')
        text_file.write('*'*60)
        text_file.write('\n\nn_estimators = %i' % (n_estimators))
        text_file.write('\nstop_time = %i' % (stop_time))
        for i in range(0, 10):
            fit_time_aux = int(round(time.time() * 1000))
            csv_file = 'diversity_results_iter_' + str(i) + '_' + time.strftime("%H_%M_%S", time.localtime(time.time())) + '.csv'
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=i*10)
            ensemble_classifier = DiversityEnsembleClassifier(algorithms=alg,
                                                              len_X = len(X_train),
                                                              population_size=n_estimators, 
                                                              max_epochs=stop_time,
                                                              random_state=i*10)
            print('\n\nIteration = ',i)
            text_file.write("\n\nIteration = %i" % (i))
            ensemble, classifiers_fitness = ensemble_classifier.fit(X_train, y_train, csv_file)
            ensemble_classifier.fit_ensemble(X_train, y_train, ensemble, classifiers_fitness)
            fit_total_time = (int(round(time.time() * 1000)) - fit_time_aux)
            text_file.write("\n\nDEC fit done in %i" % (fit_total_time))
            text_file.write(" ms")
            predict_aux = int(round(time.time() * 1000))
            y_pred = ensemble_classifier.predict(X_test)
            predict_total_time = (int(round(time.time() * 1000)) - predict_aux)
            text_file.write("\n\nDEC predict done in %i" % (predict_total_time))
            text_file.write(" ms")
            accuracy = accuracy_score(y_test, y_pred)
            total_accuracy.append(accuracy)
            try: 
                f1 = f1_score(y_test, y_pred)
                total_f1.append(f1)
            except: pass
            try: 
                precision = precision_score(y_test, y_pred)
                total_precision.append(precision)
            except: pass
            try: 
                recall = recall_score(y_test, y_pred)
                total_recall.append(recall)
            except: pass
            try: 
                auc = roc_auc_score(y_test, y_pred)
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
        text_file.write("\n\nAverage Accuracy = %f\n" % (statistics.mean(total_accuracy)))
        print("DEBUG >>>>>> ", total_accuracy)
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
        opts, args = getopt.getopt(argv,"h:i:o:e:s:",["ifile=","ofile=","enumber=","stoptime="])
    except getopt.GetoptError:
        print('diversity_ensemble.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time>')
        sys.exit(2)
    if opts == []:
        print('diversity_ensemble.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('diversity_ensemble.py -i <inputfile> -o <outputfile> -e <n_estimators> -s <stop_time>')
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
    
    now = time.time()
    struct_now = time.localtime(now)
    mlsec = repr(now).split('.')[1][:3]
    start_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
    print('\nStart time = ', start_time)
    print('\n')
    
    if inputfile == "iris":
        dataset = datasets.load_iris()
        print('Runing Diversity-based Ensemble Classifier...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time))
    elif inputfile == "breast":
        dataset = datasets.load_breast_cancer()
        print('Runing Diversity-based Ensemble Classifier...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time))
    elif  inputfile == "wine":
        dataset = datasets.load_wine()
        print('Runing Diversity-based Ensemble Classifier...')
        compare_results(data=dataset.data, 
                        target=dataset.target, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time))
    else:
        le = LabelEncoder()
        dataset = pd.read_csv(inputfile)
        dataset.iloc[:, -1] = le.fit_transform(dataset.iloc[:, -1])
        print('Runing Diversity-based Ensemble Classifier...')
        compare_results(data=dataset.iloc[:, 0:-1].values, 
                        target=dataset.iloc[:, -1].values, 
                        n_estimators=int(n_estimators), 
                        outputfile=outputfile, 
                        stop_time=int(stop_time))
        
    now = time.time()
    struct_now = time.localtime(now)
    mlsec = repr(now).split('.')[1][:3]
    end_time = time.strftime("%Y-%m-%d %H:%M:%S.{} %Z".format(mlsec), struct_now)
    print('\nEnd time = ', end_time)
    print('\nIt is finished!')

if __name__ == "__main__":
    main(sys.argv[1:])