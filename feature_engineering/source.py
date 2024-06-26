import numpy as np
import pandas as pd

import sklearn
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

from matplotlib import pyplot as plt

import seaborn as sns

from copy import deepcopy

class Dataset:
    def __init__(self):
        pass

    def load_csv(self, filename, **kwargs):
        """
        read CSV dataset file and return as pandas.DataFrame
        filename: filename of dataset file
        """
        
        return pd.read_csv(filename, **kwargs)
    
    def save_csv(self, df, filename, **kargs):
        df.to_csv(filename, **kargs)
    
    def load_dataset(data_name):
        df = eval('sklearn.datasets.'+data_name)()
        data = pd.DataFrame(df.data, columns=df.feature_names)
        target = pd.DataFrame(df.target, columns='target')
    
    def scale(self, df, start=0, end=-1):
        return df[start:end]

    

class DataCleaning:
    def __init__(self):
        pass
    
    def delete_unique(self, df):
        counts = df.nunique()
        to_del = [i for i, v in enumerate(counts) if (float(v)/df.shape[0]*100) > 1]
        return self.delete_by_columns(df, to_del)
    
    def delete_duplicate(self, df):
        df.drop_duplicates(inplace=True)
        return df
    
    def delete_missing(self, df_feature):
        dataset = df_feature.replace(0, np.nan)
        dataset.dropna(inplace=True)
        return dataset
    
    def delete_by_columns(self, df, columns):
        temp = deepcopy(df)
        temp.drop(columns, axis=1, inplace=True)
        return temp


class DataSelection:
    def __init__(self):
        pass

    def recursive(self, feature, target, cv=2):
        """
        Feature ranking with recursive feature elimination and cross-validated selection of the best number of features

        This function returns selected and non-selected columns using recurcive elimination model

        example return output
        ___________________________________________________
        |                     Columns  Selection  Ranking |
        | 0               mean radius       True        1 |
        | 1              mean texture       True        1 |
        | 2            mean perimeter       True        1 |
        | 3                 mean area      False        3 |
        | 4           mean smoothness       True        1 |
        | 5          mean compactness       True        1 |
        | 6            mean concavity       True        1 |
        | 7       mean concave points       True        1 |
        | 8             mean symmetry       True        1 |
        | 9    mean fractal dimension       True        1 |
        | 10             radius error       True        1 |
        | 11            texture error       True        1 |
        | 12          perimeter error       True        1 |
        | 13               area error       True        1 |
        | 14         smoothness error       True        1 |
        | 15        compactness error       True        1 |
        | 16          concavity error       True        1 |
        | 17     concave points error       True        1 |
        | 18           symmetry error       True        1 |
        | 19  fractal dimension error       True        1 |
        | 20             worst radius       True        1 |
        | 21            worst texture       True        1 |
        | 22          worst perimeter       True        1 |
        | 23               worst area      False        2 |
        | 24         worst smoothness       True        1 |
        | 25        worst compactness       True        1 |
        | 26          worst concavity       True        1 |
        | 27     worst concave points       True        1 |
        | 28           worst symmetry       True        1 |
        | 29  worst fractal dimension       True        1 |
        |_________________________________________________|
        """
        X               = feature.iloc[:, :]
        Y               = target.iloc[:, :]
        lin_reg         = LinearRegression()
        rfe_mod         = RFECV(lin_reg, cv=cv)
        values          = rfe_mod.fit(X, Y)
        rs = pd.DataFrame()
        rs['Columns'] = X.columns
        rs['Selection'] = values.support_
        rs['Ranking'] = values.ranking_
        return rs

    def embedded(self, feature, target, alpha=1):
        """
        example return output
        _____________________________________
        |     Columns  Coefficient Estimate |
        | 0      CRIM             -0.045348 |
        | 1        ZN              0.048966 |
        | 2     INDUS             -0.000000 |
        | 3      CHAS              0.000000 |
        | 4       NOX             -0.000000 |
        | 5        RM              0.843618 |
        | 6       AGE              0.029624 |
        | 7       DIS             -0.560082 |
        | 8       RAD              0.287359 |
        | 9       TAX             -0.016186 |
        | 10  PTRATIO             -0.838047 |
        | 11        B              0.007720 |
        | 12    LSTAT             -0.802309 |
        |___________________________________|
        """
        X = feature.iloc[:, :]
        Y = target.iloc[:, :]
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, Y)
        lasso_coeff = pd.DataFrame()
        lasso_coeff['Columns'] = X.columns
        lasso_coeff['Coefficient Estimate'] = pd.Series(lasso.coef_)
        return lasso_coeff
    
    def random_forest(self, feature, target, n_estimators=100, random_state=15, plot=False, img_name=''):
        """
        example return output
        ____________________________________
        |              Columns  Importance |
        | 0  sepal length (cm)    0.006579 |
        | 1   sepal width (cm)    0.011457 |
        | 2  petal length (cm)    0.443272 |
        | 3   petal width (cm)    0.538692 |
        |__________________________________|
        """
        X = feature.iloc[:, :]
        Y = target.iloc[:, :]
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, Y)
        importance = rf.feature_importances_
        rs = pd.DataFrame()
        rs['Columns'] = X.columns
        rs['Importance'] = importance

        if plot:
            forest_importance = pd.Series(importance, index=X.columns)
            fig, ax = plt.subplots()
            f = forest_importance.plot.bar()
            ax.set_title("Feature importances")
            ax.set_ylabel("Mean decrease in impurity")
            f.figure.savefig(img_name, bbox_inches='tight')
        return rs

    def varieance_threshold(self, df, threshold):
        """
        example return output
        __________________________
        |     Columns  Threshold |
        | 0      CRIM       True |
        | 1        ZN       True |
        | 2     INDUS       True |
        | 3      CHAS      False |
        | 4       NOX      False |
        | 5        RM      False |
        | 6       AGE       True |
        | 7       DIS      False |
        | 8       RAD       True |
        | 9       TAX       True |
        | 10  PTRATIO      False |
        | 11        B       True |
        | 12    LSTAT       True |
        | 13   Target       True |
        |________________________|
        """
        var_thres = VarianceThreshold(threshold=threshold)
        var_thres.fit(df)
        # support = var_thres.get_support()
        rs = pd.DataFrame()
        rs['Columns'] = df.columns
        rs['Threshold'] = pd.Series(var_thres.get_support())
        return rs

    def kbest(self, feature, target, k=2):
        X = feature.iloc[:, :]
        Y = target.iloc[:, :]
        selector = SelectKBest(score_func=chi2, k=k)
        selection = selector.fit(X, Y)

        rs = pd.DataFrame()
        rs['Columns'] = X.columns
        rs['Scores'] = selection.scores_
        return rs
    
    def percentile(self, feature, target):
        """
        example return output
        __________________________________________________
        |                          Columns        Scores |
        | 0                        alcohol      5.445499 |
        | 1                     malic_acid     28.068605 |
        | 2                            ash      0.743381 |
        | 3              alcalinity_of_ash     29.383695 |
        | 4                      magnesium     45.026381 |
        | 5                  total_phenols     15.623076 |
        | 6                     flavanoids     63.334308 |
        | 7           nonflavanoid_phenols      1.815485 |
        | 8                proanthocyanins      9.368283 |
        | 9                color_intensity    109.016647 |
        | 10                           hue      5.182540 |
        | 11  od280/od315_of_diluted_wines     23.389883 |
        | 12                       proline  16540.067145 |
        |_________________________________________________
        """
        X = feature.iloc[:, :]
        Y = target.iloc[:, :]
        selector = SelectPercentile(chi2)
        selection = selector.fit(X, Y)
        rs = pd.DataFrame()
        rs['Columns'] = X.columns
        rs['Scores'] = selection.scores_
        return rs


class DataPlot:
    def __init__(self):
        pass

    def correlation(self, df, img_name):
        corr_mat = df.corr()
        f = sns.heatmap(corr_mat, annot=True)
        plt.savefig(img_name, bbox_inches='tight')
    




