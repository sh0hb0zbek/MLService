import sklearn
from sklearn import cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
COLORS = list(mcolors.CSS4_COLORS.keys())

import argparse
import pickle
from time import time
from datetime import timedelta


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


def dataset(data):
    """
    data: <pandas.DataFrame>
        given dataset
    This function return <numpy.ndarray> type data
    """
    cols = data.columns
    dataset_ = data.iloc[:].values
    return dataset_, cols


class ClusterModel:
    def __init__(self, model_name):
        """
        model_name: <str>
            name of cluster model in sklearn.cluster
        AffinityPropagation
        AgglomerativeClustering
        Birch
        BisectingKMeans
        KMeans
        MeanShift
        MiniBatchKMeans
        OPTICS
        SpectralBiclustering
        SpectralClustering
        """
        self.model_name = model_name
    
    def fit(self,X, n_clusters):
        """
        X: <numpy.ndarray>
            dataset for fitting the model
        """
        st_time = time()
        self.model = eval("cluster." + self.model_name)(n_clusters=n_clusters).fit(X)
        self.time_fit = timedelta(time() - st_time)
    
    def predict(self, test):
        """
        test: <numpy.ndarray>
            test dataset used for prediction
        """
        pred = self.model.predict(test)
        return pred
    
    def save(self, filename=None):
        if filename is None:
            filename = self.model_name + '.sav'
        pickle.down(self.model, open(filename, 'wb'))
    
    def plot(self, X, x_label=None, y_label=None):
        """
        X: <numpy.ndarray>
            dataset for visualization
        """
        labels = self.model.labels_
        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], marker='.', color=COLORS[labels[i]], markersize=10)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


def argparse_cluster_models():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',   type=str,   required=True,                help='Name of Sci-Kit Linear Models.')
    parser.add_argument('--n_clusters',   type=int,   required=True,                help='Number of clusters.')
    parser.add_argument('--filepath',     type=str,   required=True,                help='Path for CSV dataset file.')
    parser.add_argument('--scaler_type',  type=str,   required=False, default=None, help='Name of scaler of Sci-Kit Preprocessing.')
    parser.add_argument('--encoder_type', type=str,   required=False, default=None, help='Name of encoder of Sci-Kit Preprocessing.')
    parser.add_argument('--save',         action='store_true',                      help='Whether or not to save the fitted model')
    parser.add_argument('--filename',     type=str,   required=False, default=None, help='File name to save the fitted model')
    return parser.parse_args()