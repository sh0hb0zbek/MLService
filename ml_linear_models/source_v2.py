# data loading
# data pre-processing
# model fitting, evaluation and result visualization
# model saving

from random import random
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn.linear_model
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import argparse
from time import time
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')


# data loading
"""
    In this process, dataset will be loaded and ends by returning [X], [y] datasets
"""

def data_filetype(filepath):
    """
    This function returns a path of dataset file and returns what type of file it is
    e.g. .csv file, kaggle, ...
    """
    # TODO: ...
    return 'csv'


def datafile_loader(filepath, filetype):
    """
    This function returns a dataset in form of pandas.DataFrame.
    """
    if data_filetype(filetype) == 'csv':
        return pd.read_csv(filepath)
    if data_filetype(filetype) == 'kaggle':
        pass


def data_sorting(dataset, sort=False):
    """
    This function displays names of available features by indexes.
    User should sort the features for usage of model training,
    as well as the target variable by specifying indexes.
    If sort is False as default, the last feature will be considered
    as target variable and rest of them will be feature variables
    
    The function returns two numpy.ndarray datasets:
        features (X) and target (y) respectively
    """
    cols = dataset.columns
    dataset = dataset.iloc[:].values
    features = list()
    if not sort:
        X = dataset[:, :-1]
        y = dataset[:,  -1]
    else:
        print('Sort the dataset')
        for i, feature in enumerate(cols):
            print(i, feature)
        target   = int(input('Target Variable Index:       '))
        features = [int(x) for x in input('Feature Variables Index(es): ').split()]
        X = dataset[:, features]
        y = dataset[:, target]
    return X, y


# data pre-processing
"""
    In this process, [X] and [y] dataset will be encoded if needed
    and splitted into train-test datasets
"""

def data_split(X, y, scaler_type=None, **args):
    """
    This function encodes non-numeric features (string) 
    and split the datasets into train-test parts

    args: is a dictionary which contains arguments of train_test_split() method
    """
    X = one_hot_encoder(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, **args)

    if scaler_type:
        x_train, x_test = scaler(scaler_type, x_train, x_test)
    
    return x_train, x_test, y_train, y_test


def one_hot_encoder(dataset):
    """
    dataset: <np.ndarray>
        the dataset array which string features are to be encoded
    
    return: encoded dataset
    """
    string_cloumns = list()
    for i in range(dataset.shape[1]):
        if isinstance(dataset[0, i], str):
            string_cloumns.append(i)
    if string_cloumns == list():
        return dataset
    encoder = ColumnTransformer(
        [('encoder', OneHotEncoder(sparse=False), string_cloumns)], remainder='passthrough')
    
    return np.array(encoder.fit_transform(dataset), dtype=float)


def scaler(scaler_type, train, test):
    """
    scaler_type: <str>
        type of scaler in sklearn.preprocessing
    train:       <numpy.ndarray>
        train dataset that needs to be scaled
    test:        <numpy.ndarray>
        test  dataset that needs to be scaled
    """
    sc = eval('sklearn.preprocessing.'+scaler_type)()
    train = sc.fit_transform(train)
    test  = sc.transform(test)
    return train, test


# model [fitting, predicting, evaluating, visualizing, saving]
class LinearModel():
    def __init__(self, model_name):
        """
        model_name: <str>
            name of linear model in sklearn.linear_model
        """
        self.model_name = model_name
        self.isRegressor()
        self.time = 0
    
    def isRegressor(self):
        regressor_models = [
            'ARDRegression',
            'BayesianRidge',
            'ElasticNet',
            'ElasticNetCV',
            'GammaRegressor',
            'HuberRegressor',
            'Lars',
            'LarsCV',
            'Lasso',
            'LassoCV',
            'LassoLars',
            'LassoLarsCV',
            'LassoLarsIC',
            'LinearRegression',
            'OrthogonalMatchingPursuit',
            'PassiveAggressiveRegressor',
            'PoissonRegressor',
            'QuantileRegressor',
            'RANSACRegressor',
            'Ridge',
            'RidgeCV',
            'SGDRegressor',
            'TheilSenRegressor',
            'TweedieRegressor']

        classifier_models = [
            'LogisticRegression',
            'LogisticRegressionCV',
            'PassiveAggressiveClassifier',
            'Perceptron',
            'RidgeClassifier',
            'RidgeClassifierCV',
            'SGDClassifier']
        
        if self.model_name in regressor_models:
            self.model_type = 'regressor'
        elif self.model_name in classifier_models:
            self.model_type = 'classifier'
        else:
            print('[ERROR] Selected model is not available!')
            exit(1)
    
    def fit(self, x_train, y_train):
        """
        x_train:    <numpy.ndarray>
            train dataset used for fitting the model
        
        y_train:    <numpy.ndarray>
            target dataset used fot fitting the model
        """
        st_time = time()
        self.model = eval('sklearn.linear_model.' + self.model_name)()
        self.model.fit(x_train, y_train)
        self.time_fit = timedelta(seconds=time() - st_time)

    def predict(self, test):
        """
        test: <numpy.ndarray>
            test dataset used for evaluation of the model
        """
        pred = self.model.predict(test)
        return pred

    def evaluate(self, x_test, y_test, eval_type=None):
        if eval_type is None:
            if self.model_type == 'regressor':
                eval_type = 'mean_squared_error'
            else:
                eval_type = 'confusion_matrix'
        pred = self.predict(x_test)
        if isinstance(eval_type, list):
            rs = list()
            for e in eval_type:
                rs.append(eval('sklearn.metrics.'+e)(y_test, pred))
            return rs
        return eval('sklearn.metrics.'+eval_type)(y_test, pred)
    
    # model saving
    def save(self, filename=None):
        if filename is None:
            filename = self.model_name + '.sav'
        pickle.down(self.model, open(filename, 'wb'))
    
    # result visualization
    def plot(self, X, target):
        print(f'[{self.model_name}]')
        if self.model_type == 'regressor':
            self.plot_reg(X, target)
        else:
            self.plot_cls(*self.evaluate(X, target, [
                'confusion_matrix',
                'accuracy_score']))
        print('Time to fit model:', self.time_fit, '\n-----------------------------------------------------------------\n')
    
    def plot_reg(self, X, target):
        pred = self.predict(X)
        plt.scatter(x=X, y=target, color='red')
        plt.plot(x=X, y=pred, color='blue')

    def plot_cls(self, cm, acc_score):
        print(f'Accuracy:  {acc_score}')
        print('Confusion Matrix:')
        plt.figure(figsize=(15, 7))
        ax = sns.heatmap(cm, annot=True, annot_kws={'size': 16}, fmt='.3g')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        plt.show()


# argument parsers
def argparse_linear_models():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',   type=str,   required=True,                help='Name of Sci-Kit Linear Models.')
    parser.add_argument('--filepath',     type=str,   required=True,                help='Path for CSV dataset file.')
    parser.add_argument('--test_size',    type=float, required=False, default=0.3,  help='The propotion of the dataset to include in the test.')
    parser.add_argument('--train_size',   type=float, required=False, default=None, help='The propotion of the dataset to include in the train.')
    parser.add_argument('--random_state', type=int,   required=False, default=None, help='Controls the shuffling applied to the data before applying the split.')
    parser.add_argument('--shuffle',      action='store_true',                      help='Whether or not to shuffle the data before splitting.')
    parser.add_argument('--sort',         action='store_true',                      help='Whether or not to sort features of data')
    parser.add_argument('--scaler_type',  type=str,   required=False, default=None, help='Name of scaler of Sci-Kit Preprocessing.')
    parser.add_argument('--save',         action='store_true',                      help='Whether or not to save the fitted model')
    parser.add_argument('--filename',     type=str,   required=False, default=None, help='File name to save the fitted model')
    return parser.parse_args()