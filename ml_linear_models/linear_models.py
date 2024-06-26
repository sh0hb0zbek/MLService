import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import plot_confusion_matrix
from time import time
from datetime import timedelta

def data_loader(filepath, test_size=0.3, train_size=None,
                random_state=None, shuffle=True, encode=False):
    """
    filepath:     <str>
        path of .csv dataset file
    
    test_size:    <float> [0.0, 1.0], default=0.3
        the proportion of the dataset to include in the test
    
    train_size:   <float> [0.0, 1.0], default=None
        the proportion of the dataset to include in the train
        * if train_size is given, test_size will not be used
    
    random_state: <int>,  default=None
        controls the shuffling applied to the data 
        before applying the split
    
    shuffle:      <bool>, default=True
        whether or not to shuffle the data before splitting
    
    encoded:      <bool>, default=True
    
    """
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    encoder = None

    if encode:
        X, _ = one_hot_encoder(X)
        Y, encoder = one_hot_encoder(Y)

    if train_size is None:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    else:
        X_train, X_test, Y_train, y_test = train_test_split(
            X, Y, train_size=train_size, random_state=random_state, shuffle=shuffle)
    return X_train, X_test, Y_train, Y_test, encoder


def scaler(scaler_type, train, test):
    """
    scaler_type: <str>
        type of scaler in sklearn.preprocessing
    train:       <numpy.ndarray>
        train dataset that needs to be scaled
    test:        <numpy.ndarray>
        test  dataset that needs to be scaled
    """
    string_mapping = dict()
    for i in range(train.shape[1]):
        if isinstance(train[:, i][0], str):
            classes = list(set(train[:, i]))
            mapping = dict()
            for j in range(len(classes)):
                mapping[classes[j]] = j
            string_mapping[i] = mapping
            for k in range(train.shape[0]):
                train[k][i] = mapping[train[k][i]]
            for l in range(test.shape[0]):
                test[l][i] = mapping[test[l][i]]
    sc = eval('sklearn.preprocessing.'+scaler_type)()
    train = sc.fit_transform(train)
    test  = sc.transform(test)
    return train, test


def linear_model(model_type):
    """
    model_type: <str>
        name of linear model in sklearn.linear_model
    """
    return eval('sklearn.linear_model.' + model_type)


def one_hot_encoder(dataset):
    """
    dataset: <np.ndarray>
        the dataset array which string features are to be encoded
    
    return: encoded dataset, encoder
        there can be a need to inverse the encoded data,
        that's why encoder is returned as well
    """
    string_cloumns = list()
    for i in range(dataset[0]):
        if isinstance(dataset[0, i], str):
            string_cloumns.append(i)
    if string_cloumns == list():
        return dataset, None
    encoder = ColumnTransformer(
        [('encoder', OneHotEncoder(sparse=False), string_cloumns)], remainder='passthrough')
    
    return np.array(encoder.fit_transform(dataset), dtype=float), encoder


def model_fit(model_type, train, target):
    """
    model_type: <str>
        name of linear model in sklearn.linear_model
    
    train:      <numpy.ndarray>
        train dataset used for fitting the model
    
    target:     <numpy.ndarray>
        target dataset used fot fitting the model
    """
    model = linear_model(model_type)()
    return model.fit(train, target)


def predict(model, test):
    """
    model:
        trained linear model
    
    test: <numpy.ndarray>
        test dataset used for evaluation of the model
    """
    return model.predict(test)


def model_eval(model, x_test, y_test, eval_type='accuracy_score'):
    pred = predict(model, x_test)
    return eval('sklearn.metrics.'+eval_type)(y_test, pred)


def plot(plot_type, **args):
    if plot_type in ['scatter', 'plot']:
        eval('plt.'+plot_type)(args['x'], args['y'], color=args['color'])


def exec_time(st_time):
    return str(timedelta(seconds=time() - st_time))


def linear_model_main(model_type, filepath, test_size=0.3, train_size=None,
                      random_state=None, shuffle=True, encode=False,
                      scaler_type=None, eval_type=None, plot_type=None):
    st_time = time()

    # data preparation
    x_train, x_test, y_train, y_test, encoder = data_loader(
        filepath, test_size, train_size, random_state, shuffle, encode)
    
    # scaling
    if scaler_type:
        x_train, x_test = scaler(scaler_type, x_train, x_test)
    
    # model fitting
    model = model_fit(model_type, x_train, y_train)
    
    # evaluation
    if eval_type:
        accuracy = model_eval(model, x_test, y_test, eval_type)
        print(f'{eval_type}:\n{accuracy}')

    print(f'Execution_time:', exec_time(st_time))
    
    # plotting
    if plot_type:
        if plot_type != 'confusion_matrix':
            pred_test = predict(model, x_test)
            plot('scatter', x=x_test, y=y_test, color='red')
            plot('plot', x=x_test, y=pred_test, color='blue')
            plt.show()
        else:
            plot_confusion_matrix(model, x_test, y_test)
            plt.show()

    if encode:
        return model, encoder
    
    return model