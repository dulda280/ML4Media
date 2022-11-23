import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import feature_selection
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)

    plt.plot(X)
    plt.show()

    pct = Perceptron(tol=1e-3, random_state=0)
    pct.fit(X, y)
    print(pct.score(X, y))


    print(pct.get_params())
    parameters = {'alpha': [0.0001, 0.5, 0.000001], 'eta0': [1.0, 2.0, 5.0, 7.5, 10.0], 'l1_ratio': [0.05, 0.01, 0.5],  'tol': [0.001, 0.01, 0.1, 1.0], 'validation_fraction': [0.1, 0.2, 0.4, 0.9]}
    pct2 = Perceptron(tol=1e-3, random_state=0)
    
    clf = GridSearchCV(pct2, parameters)
    clf.fit(X, y)

    print(f'Best estimator: {clf.best_estimator_}')
    print(f'Params: {parameters}')
    print(f'Best params: {clf.best_params_}')
    print(f'Best score: {clf.best_score_}')