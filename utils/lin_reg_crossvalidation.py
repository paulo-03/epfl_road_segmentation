"""
Helper functions to perform cross validation on logistic regression
"""

import warnings

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


def lin_reg_best_params(penalties, class_weights, x, y):
    """
    Perform cross validation to find the best parameters for a logistic regression model

    :param penalties: list of penalties to try
    :param class_weights: list of class weights to try
    :param x: the features
    :param y: the labels
    :return: the best parameters and the best f1 score
    """
    # disable warnings for readability
    warnings.filterwarnings('ignore')

    best_score = -1
    best_params = None

    for penalty in penalties:
        if penalty is None:
            solvers = ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            cs = [1]
            l1_ratios = [0]
        elif penalty == 'l1':
            solvers = ['liblinear', 'saga']
            cs = np.logspace(-8, 10, 8)
            l1_ratios = [1]
        elif penalty == 'l2':
            solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            cs = np.logspace(-8, 10, 8)
            l1_ratios = [0]
        else:
            solvers = ['saga']
            cs = np.logspace(-8, 10, 8)
            l1_ratios = np.linspace(0, 1, 10)

        for solver in solvers:
            for C in cs:
                for class_weight in class_weights:
                    for l1_ratio in l1_ratios:
                        # evaluate the model
                        logreg = linear_model.LogisticRegression(C=C,
                                                                 penalty=penalty,
                                                                 class_weight=class_weight,
                                                                 solver=solver,
                                                                 l1_ratio=l1_ratio,
                                                                 max_iter=1000)
                        f1_score = cross_val_score(logreg, x, y, cv=10, scoring="f1").mean()
                        # update best score and parameters if necessary
                        if f1_score > best_score:
                            best_score = f1_score
                            best_params = {'penalty': penalty,
                                           'C': C,
                                           'solver': solver,
                                           'class_weight': class_weight,
                                           'l1_ratio': l1_ratio}
    warnings.filterwarnings("default")
    return best_params, best_score
