from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
class CustomSVC(BaseEstimator):
    def __init__(self, C=1.0, n_components=None, random_state=None):
        self.steps = [('pca', PCA(n_components=n_components, random_state=random_state)),
                      ('clf', SVC(C=C, random_state=random_state))]
        self.pipe_lr = Pipeline(self.steps)
    
    def fit(self, X, y):
        self.pipe_lr.fit(X, y)

    def predict(self, X):
        return self.pipe_lr.predict(X)