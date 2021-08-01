import math
import collections

def gen_members(dims):
    n_samples = int(math.sqrt(dims[0]))

    members = {
            'sklearn.neighbors.KNeighborsClassifier': {'n_neighbors':[1, 3, 7, n_samples], 'weights':['uniform', 'distance']},
            'sklearn.neighbors.RadiusNeighborsClassifier': {'radius':[0.5, 0.7, 1.0], 'weights':['uniform', 'distance'], 'outlier_label':[2]},
            'sklearn.neighbors.NearestCentroid': {},
            'sklearn.svm.SVC': {'C':[1,1000],'gamma':[0.0001, 0.0005, 0.001]},
            'sklearn.svm.LinearSVC': {'C':[1,1000],'max_iter':[100]},
            'sklearn.gaussian_process.GaussianProcessClassifier': {},
            'sklearn.tree.DecisionTreeClassifier': {'min_samples_leaf':[1, 5], 'max_depth':[5, n_samples]},
            'sklearn.tree.ExtraTreeClassifier': {'min_samples_leaf':[1, n_samples], 'max_depth':[1, n_samples]},
            'sklearn.naive_bayes.GaussianNB': {},
            'sklearn.naive_bayes.BernoulliNB': {},
            'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': {},
            'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis': {},
            'sklearn.linear_model.RidgeClassifier': {'alpha':[1.0, 1000.0],'max_iter':[100]},
            'sklearn.linear_model.LogisticRegression': {'C':[1, 1000], 'max_iter':[100], 'solver':['saga'], 'tol':[1e-5]},
            'sklearn.linear_model.PassiveAggressiveClassifier': {'C':[1, 1000], 'max_iter':[100]},
            'sklearn.linear_model.SGDClassifier': {'alpha':[1e-5, 1e-2], 'max_iter':[100]},
            'sklearn.linear_model.Perceptron': {'alpha':[1e-5, 1e-2], 'max_iter':[100]},
            'sklearn.neural_network.MLPClassifier': {'alpha':[1e-5, 1e-2], 'max_iter':[100]}
    }
    
    return collections.OrderedDict(members)