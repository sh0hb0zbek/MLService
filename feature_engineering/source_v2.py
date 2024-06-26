import numpy as np
import pandas as pd

import sklearn
from sklearn import feature_selection
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from scipy import stats

import phik

import plotly.express as px

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
    
    def load_dataset(self, data_name):
        df = eval('datasets.'+data_name)()
        data = pd.DataFrame(df.data, columns=df.feature_names)
        target = pd.DataFrame(df.target, columns='target')
    
    def scale(self, df, start=0, end=-1):
        return df[start:end]
    
    def concat(self, df1, df2, axis=0, sort=False):
        """
        merge two datasets into one
        """
        return pd.concat([df1, df2], ignore_index=True, axis=axis, sort=sort)


class DataExploartion:
    def __init__(self):
        pass

    def shape(self, df):
        return df.shape
    
    def info(self, df):
        return df.info()
    
    def columns(self, df):
        return df.columns
    
    def index(self, df):
        return df.index
    
    def describe(self, df):
        return df.describe()
    
    def head(self, df, n=5):
        return df.head(n)
    
    def tail(self, df, n=5):
        return df.tail(n)
    
    def sort_column(self, df, columns):
        return df.sort_values([columns], ascending=True)
    
    def value_counts(self, df, column):
        return df[column].value_counts()
    

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
        return df.drop(columns, axis='columns', inplace=True)

    def outlier(self, df, min_max=False, selection=None, dtypes=None, distict_value=None, z=None, quantile=None, column=None):
        """
        min_max        <bool>  : removes rows containing minimum and maximum values in specific column
                                * column name must be given
        selection      <list>  : removes selected rows, list contains index value of rows to remove
        dtypes         <list>  : if columns' data type is the same as [dtype], that column will be removed
                                * list contains string name of dtype
        distinct_value <list>  : removes rows containing given distinct value list
                                * column name containing given distinct value must be given
        z              <float> : removes all rows that have outliers in at least one column
        quantile       <list>  : filters with quantile range in specific column and removes ouliers
                                * column name must be given
                                quantile=[lower quantile, higher quantile]
        """
        if min_max is not None and column is not None:
            df_copy = deepcopy(df)
            df_copy = df_copy.drop(df_copy[column].idxmax())
            df_copy = df_copy.drop(df_copy[column].idxmin())
            return df_copy
        if selection is not None:
            return df.drop(labels=selection, axis=0)
        if dtypes is not None:
            df_copy = deepcopy(df)
            for dtype in dtypes:
                df_copy = df_copy.select_dtypes(exclude=[dtype])
            return df_copy
        if distict_value is not None and column is not None:
            df_copy = deepcopy(df)
            for value in distict_value:
                df_copy = df_copy.drop(df_copy.loc[df_copy[column]==value].index, inplace=True)
            return df_copy
        if z is not None:
            return df[(np.abs(stats.zscore(df)) < z).all(axis=1)]
        if quantile is not None and column is not None:
            return df[(df[column] < df[column].quantile(quantile[1]) & df[column] > df[column].quantile(quantile[0]))]


class DataExtraction:
    def __init__(self):
        pass

    


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

    def variance_threshold(self, df, threshold):
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

    def change_values(self, df, change_value=None, spec_value=None, negative=None, missing=None, zero=None, interpolate=False, drop_row=False, drop_column=False):
        if missing is not None:
            if interpolate:
                """
                input dataset
                        A       B       C       D|
                ---------------------------------|
                0    12.0     NaN    20.0    14.0|
                1     4.0     2.0    16.0     3.0|
                2     5.0    54.0     NaN     NaN|
                3     NaN     3.0     3.0     NaN|
                4     1.0     NaN     8.0     6.0|

                output data
                        A       B       C       D|
                ---------------------------------|
                0    12.0     NaN    20.0    14.0|
                1     4.0     2.0    16.0     3.0|
                2     5.0    54.0     9.5     4.0|
                3     3.0     3.0     3.0     5.0|
                4     1.0     3.0     8.0     6.0|
                
                values in the first row cannot get filled as the direction of filling of values is forward
                and there is no previous valu which could have been used in interpolation
                """
                return df.interpolate(method='linear', limit_direction='forward')
            elif drop_row:
                """
                drops rows with at least one Nan value (Null value)
                """
                return df.dropna()
            elif drop_column:
                """
                drops columns with at least one Nan value (Null value)
                """
                return df.dropna(axis=1)
            else:
                return df.replace(to_replace=np.nan, value=change_value, inplace=True)
        if negative is not None:
            return df.clip(lower=change_value)
        if zero is not None:
            return df.replace(to_resplace=0, value=change_value, inplace=True)
        if spec_value is not None:
            return df.replace(to_replace=spec_value, value=change_value, inplace=True)

class DataPlot:
    def __init__(self):
        pass

    def correlation(self, df, img_name, corr_type=None):
        if corr_type in ['spearman', 'pearson', 'kendall']:
            corr_mat = df.corr(method=corr_type)
        elif corr_type == 'phik':
            corr_mat = df.phik_matrix()
        f = sns.heatmap(corr_mat, annot=True)
        plt.savefig(img_name, bbox_inches='tight')
    
    def pca(self, df, img_name):
        pca_ = PCA()
        components = pca_.fit_transform(df[df.columns])
        labels = {
            str(i): f'PC {i+1} {var:.1f}%'
            for i, var in enumerate(pca_.explained_variance_ratio_ * 100)
        }
        fig = px.scatter_matrix(
            components,
            labels=labels,
            color=df[df.columns[-1]] 
        )
        fig.update_traces(diagonal_visible=False)
        fig.write_image(img_name)


class Processing:
    def __init__(self):
        pass

    def scaler(self, df, scaler_type, method='yeo-johnson', quantile_range=(25.0, 75.0), output_distribution='uniform', norm='l2'):
        """
        scaler_type:
            - MinMaxScaler
            - MaxAbsScaler
            - StandardScaler
            - RobustScaler
                - quantile_range: <tuple> (lower_quantile, higher_quantile)
            - Quantile Transformer
                - output_distribution: 'uniform' or 'normal'
                    - 'uniform' - 'uniform  pdf'
                    - 'normal'  - 'gaussion pdf'
            - PowerTransformer
                - method: 'yeo-johnson' or 'box-cox'
            - Normalizer
                - norm: 'l1', 'l2', 'max'
            
            input df: <numpy.ndarray> or <pandas.DataFrame> 
            return  : <np.ndarray>
        """
        if scaler_type in ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler']:
            return eval(scaler_type)().fit_transform(df)
        if scaler_type == 'RobustScaler':
            return RobustScaler(quantile_range=quantile_range).fit_transform(df)
        if scaler_type == 'QuantileTransformer':
            return QuantileTransformer(output_distribution=output_distribution).fit_transform(df)
        if scaler_type == 'PowerTransformer':
            return PowerTransformer(method=method)
        if scaler_type == 'Normalizer':
            return Normalizer(norm=norm).fit_transform(df)

    
