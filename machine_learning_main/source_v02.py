# data loading
# data pre-processing
# model fitting, evaluation and result visualization
# model saving

import pandas as pd
import numpy as np
import sklearn
import sklearn.neighbors
import sklearn.tree
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn.linear_model
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import pickle
import argparse
from time import time
from datetime import timedelta
import inspect

import warnings
warnings.filterwarnings('ignore')

# available sci-kit models for supervised learning
MODELS = {                  # 57 models
    'regressor': {          # 37 models
        'linear_model': [   # 24 models
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
            'TweedieRegressor'],
        'neighbors': [      # 2 models
            'KNeighborsRegressor',
            'RadiusNeighborsRegressor'],
        'svm': [            # 3 models
            'LinearSVR',
            'NuSVR',
            'SVR'],
        'tree': [           # 2 models
            'DecisionTreeRegressor',
            'ExtraTreeRegressor'],
        'ensemble': [       # 6 models
            'AdaBoostRegressor',
            'BaggingRegressor',
            'ExtraTreesRegressor',
            'GradientBoostingRegressor',
            'HistGradientBoostingRegressor',
            'RandomForestRegressor'],
    },
    'classifier': {         # 20 models
        'linear_model': [   # 7 models
            'LogisticRegression',
            'LogisticRegressionCV',
            'PassiveAggressiveClassifier',
            'Perceptron',
            'RidgeClassifier',
            'RidgeClassifierCV',
            'SGDClassifier'],
        'neighbors': [      # 2 models
            'KNeighborsClassifier',
            'RadiusNeighborsClassifier'],
        'svm': [            # 3 models
            'LinearSVC',
            'NuSVC',
            'SVC'],
        'tree': [           # 2 models
            'DecisionTreeClassifier',
            'ExtraTreeClassifier'],
        'ensemble': [       # 6 models
            'AdaBoostClassifier',
            'BaggingClassifier',
            'ExtraTreesClassifier',
            'GradientBoostingClassifier',
            'HistGradientBoostingClassifier',
            'RandomForestClassifier'],
    }
}
        


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

class Model():
    def __init__(self, model_name=None, load=False, filepath=None, **args):
        """
        model_name: default=None    <str>
            name of linear model in sklearn.linear_model
        load:       defualt=False   <bool>
            if true, model will be loaded from file
        filepath:   default=None    <str>
            if load model from file, filepath of model file must be given
        """
        if load:
            model.load(filepath)
            return
        self.model_name = model_name
        self.model_check()
        self.model_args = dict()
        self.set_args(args)
    
    def model_check(self):
        for model_type in MODELS.keys():
            for model_family in MODELS[model_type].keys():
                if self.model_name in MODELS[model_type][model_family]:
                    self.model_type = model_type
                    self.model_f = eval('sklearn.'+model_family+'.'+self.model_name)
                    return
        print('[ERROR] Selected model is not available!')
        sys.exit(1)
    
    def set_args(self, args):
        sig = inspect.signature(self.model_f)
        keys = args.keys()
        for i in sig.parameters.keys():
            if i in keys:
                value = args[i]
            else:
                value = sig.parameters.get(i).default
            self.model_args[i] = value

    def fit(self, x_train, y_train):
        """
        x_train:    <numpy.ndarray>
            train dataset used for fitting the model
        
        y_train:    <numpy.ndarray>
            target dataset used fot fitting the model
        """
        st_time = time()
        self.model = self.model_f(**self.model_args)
        self.model.fit(x_train, y_train)
        self.time_fit = timedelta(seconds=time()-st_time)

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
                eval_type = [
                    'confusion_matrix',
                    'accuracy_score']
        pred = self.predict(x_test)
        rs = dict()
        if isinstance(eval_type, list):
            for e in eval_type:
                rs[e] = eval('sklearn.metrics.'+e)(y_test, pred)
        else:
            rs[eval_type] = eval('sklearn.metrics.'+eval_type)(y_test, pred)
        return rs, pred
    
    # model saving
    def save(self, filename=None):
        if filename is None:
            filename = self.model_name + '.sav'
        pickle.down(self.model, open(filename, 'wb'))
    
    # model loading
    def load(self, filepath):
        self.model = pickle.load(open(filepath, 'rb'))
        self.model_name = type(self.model).__name__
        for model_type in MODELS.keys():
            for model_family in MODELS[model_type].keys():
                if self.model_name in MODELS[model_type][model_family]:
                    self.model_type = model_type
                    return
        print('[ERROR] Given model is not available')
        sys.exit(1)

    # result visualization
    def plot(self, X, target, pred, filename=None):
        if self.model_type == 'regressor':
            self.plot_reg(X, target, pred)
        else:
            self.plot_cls(target, pred)
        if filename:
            plt.savefig(filename+'.png')
        plt.show()
    
    def plot_reg(self, X, target, pred):
        plt.scatter(X, target, color='red')
        plt.plot(X, pred, color='blue')
        # plt.show()

    def plot_cls(self, target, pred):
        cm = sklearn.metrics.confusion_matrix(target, pred)
        plt.figure(figsize=(15, 7))
        ax = sns.heatmap(cm, annot=True, annot_kws={'size': 16}, fmt='.3g')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        # plt.show()


def model_fit_main(
    model_name,
    dataset_filepath,
    scaler_type=None,
    test_size=0.3,
    random_state=None,
    shuffle=False,
    save_filename=None):
    """
    args = {
        'model_name':       __,
        'dataset_filepath:  __,
        'scaler_type':      __,
        'test_size':        __,
        'random_state':     __,
        'shuffle':          __,
        'save_filename':    __,
        }
    """
    # model initialization
    model = Model(model_name=model_name)

    # loading
    X, y = data_sorting(
        dataset=datafile_loader(filepath=dataset_filepath, filetype=data_filetype(filepath=dataset_filepath))
    )

    # pre-processing
    x_train, x_test, y_train, y_test = data_split(
        X, y, scaler_type=scaler_type, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    # fitting
    if model.model_type == 'classifier':
        y_train, y_test = scaler('LabelEncoder', y_train, y_test)
    model.fit(x_train=x_train, y_train=y_train)

    # saving the model
    if save_filename:
        model.save(filename=save_filename)
    
    return model.model


def model_evaluate_main(
    model_filepath,
    dataset_filepath,
    plot=False,
    plot_filename=None,):

    # load model from file
    model = Model(load=True, filepath=model_filepath)

    # load dataset
    X, y = data_sorting(
        dataset=datafile_loader(filepath=dataset_filepath, filetype=data_filetype(filepath=dataset_filepath))
    )

    # evaluate model
    results, pred = model.evaluate(X, y)
    print(f'[{model.model_name}]')
    for metrics in results.keys():
        print(f'{metrics}: {results[metrics]}')

    # plotting the graphs
    if plot:
        model.plot(X=X, target=y, pred=pred, filename=plot_filename)
    
    return model.model



def machine_learning_main(
    model_name,
    dataset_filepath,
    scaler_type=None,
    test_size=0.3,
    random_state=None,
    shuffle=False,
    plot=False,
    plot_filename=None,
    save_filename=None):
    """
    args = {
        'model_name':       __,
        'dataset_filepath': __,
        'scaler_type':      __,
        'test_size':        __,
        'random_state':     __,
        'shuffle':          __,
        'plot'              __,
        'plot_filename':    __,
        'save_filepath':    __,
    }"""
    
    # model initialization
    model = Model(model_name=model_name)

    # loading
    X, y = data_sorting(
        dataset=datafile_loader(filepath=dataset_filepath, filetype=data_filetype(filepath=dataset_filepath))
    )

    # pre-processing
    x_train, x_test, y_train, y_test = data_split(
        X, y, scaler_type=scaler_type, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    # fitting
    if model.model_type == 'classifier':
        y_train, y_test = scaler('LabelEncoder', y_train, y_test)
    model.fit(x_train=x_train, y_train=y_train)

    # evaluating
    results, pred = model.evaluate(x_test, y_test)
    print(f'[{model.model_name}]')
    for metrics in results.keys():
        print(f'{metrics}: {results[metrics]}')

    # plotting the graphs
    if plot:
        model.plot(X=x_test, target=y_test, pred=pred, filename=plot_filename)
    
    # saving the model
    if save_filename:
        model.save(filename=save_filename)


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
