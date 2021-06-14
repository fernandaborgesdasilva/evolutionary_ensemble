import math
import collections

def gen_members(dims):
    n_samples = int(math.sqrt(dims[0]))

    members = {
            'sklearn.neighbors.KNeighborsClassifier': {'n_neighbors':[1, 3, 7, n_samples], 'weights':['uniform', 'distance']},
            #'RidgeClassifier': {'alpha':[1.0, 10.0],'max_iter':[10, 100]},
            'sklearn.svm.SVC': {'C':[1, 1000]},
            #'sklearn.svm.SVC': {'C':[1, 1000],'gamma':[0.0001, 0.001]},
            #'SVC': {'C':[1, 100, 500, 1000],'gamma':[0.0001, 0.0005, 0.001]},
            #'sklearn.tree.DecisionTreeClassifier': {'min_samples_leaf':[1, 5], 'max_depth':[5, n_samples]},
            'sklearn.tree.DecisionTreeClassifier': {'min_samples_leaf':[5], 'max_depth':[5, 10]},
            #'ExtraTreeClassifier': {'min_samples_leaf':[1, n_samples], 'max_depth':[1, n_samples]},
            'sklearn.naive_bayes.GaussianNB': {},
            #'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': {},
            #'QuadraticDiscriminantAnalysis': {},
            #'BernoulliNB': {},
            'sklearn.linear_model.LogisticRegression': {'C':[1, 1000], 'max_iter':[100], 'solver':['saga'], 'tol':[1e-5]},
            #'NearestCentroid': {},
            #'sklearn.linear_model.PassiveAggressiveClassifier': {'C':[1, 1000], 'max_iter':[100]},
            'sklearn.linear_model.SGDClassifier': {'alpha':[1e-5, 1e-2], 'max_iter':[100]}
    }
    
    return collections.OrderedDict(members)

