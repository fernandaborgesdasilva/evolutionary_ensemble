from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
class CustomLogisticRegression(BaseEstimator):
    def __init__(self, C=1.0, max_iter=100, solver='lbfgs', tol=1e-4, n_components=None, random_state=None):
        self.steps = [('pca', PCA(n_components=n_components, random_state=random_state)),
                      ('clf', LogisticRegression(C=C, max_iter=max_iter, solver=solver, tol=tol, random_state=random_state))]
        self.pipe_lr = Pipeline(self.steps)
    
    def fit(self, X, y):
        self.pipe_lr.fit(X, y)

    def predict(self, X):
        return self.pipe_lr.predict(X)