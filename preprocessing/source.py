from sklearn.preprocessing import binarize
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from numpy import nan

class Encoder:
    def __init__(self):
        pass

    """TODO"""
    def binarize(self,
            X,
            threshold=0.0,
            copy=True):
        """
        equivalent function for Binarizer() without estimator API
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
        
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to binarize, element by element. scipy.sparse matrices should be in CSR or CSC
            format to avoid an un-necessary copy.

        threshold: float, default=0.0
            Feature values below or equal to this are replaced by 0, above it by 1.
            Threshold may not be less than 0 for operations on sparse matrices.

        copy: bool, default=True
            Set to False to perform inplace binarization and avoid a copy
            (if the input is already a numpy array or a scipy.sparse CSR / CSC matrix and if axis is 1).
        
        """
        return binarize(
            X,
            threshold=threshold,
            copy=copy)

    def label_binarize(self,
            X,
            classes,
            neg_label=0,
            pos_label=1,
            sparse_output=False):
        """
        function to perform the transform operation of LabelBinarizer() with fixed classes
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer

        X: array-like
            Sequence of integer labels or multilabel data to encode.

        classes: array-like of shape (n_classes,)
            Uniquely holds the label for each class.

        neg_label: int, default=0
            Value with which negative labels must be encoded.

        pos_label: int, default=1
            Value with which positive labels must be encoded.

        sparse_output: bool, default=False,
            Set to true if output binary array is desired in CSR sparse format.
        """
        return label_binarize(
            X,
            classes=classes,
            neg_label=neg_label,
            pos_label=pos_label,
            sparse_output=sparse_output)

    def Binarizer(self,
            X,
            threshold=0.0,
            copy=True):
        """
        threshold: float, default=0.0
            Feature values below or equal to this are replaced by 0, above it by 1.
            Threshold may not be less than 0 for operations on sparse matrices.

        copy: bool, default=True
            Set to False to perform inplace binarization and avoid a copy
            (if the input is already a numpy array or a scipy.sparse CSR matrix).
        """
        bin = Binarizer(
            threshold=threshold,
            copy=copy)
        bin.fit(X)
        return bin.transform(X), bin

    def KBinsDiscretizer(self,
            X,
            n_bins=5,
            encode='onehot',
            strategy='quaintile',
            dtype=None,
            subsample='warn',
            random_state=None):
        """
        n_bins: int or array-like of shape (n_features,), default=5
            The number of bins to produce. Raises ValueError if n_bins < 2.
        encode: {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
            Method used to encode the transformed result.
                'onehot': Encode the transformed result with one-hot encoding and
                    return a sparse matrix. Ignored features are always stacked to the right.
                'onehot-dense': Encode the transformed result with one-hot encoding
                    and return a dense array. Ignored features are always stacked to the right.
                'ordinal': Return the bin identifier encoded as an integer value.
        strategy: {'uniform', 'quantile', 'kmeans'}, default='quantile'
            Strategy used to define the widths of the bins.
                'uniform': All bins in each feature have identical widths.
                'quantile': All bins in each feature have the same number of points.
                'kmeans': Values in each bin have the same nearest center of a 1D k-means cluster.
        dtype: {np.float32, np.float64}, default=None
            The desired data-type for the output. If None, output dtype is consistent with input dtype.
            Only np.float32 and np.float64 are supported.
        subsample: int or None (default='warn')
            Maximum number of samples, used to fit the model, for computational efficiency.
            Used when strategy="quantile". subsample=None means that all the training samples are used
            when computing the quantiles that determine the binning thresholds. Since quantile computation
            relies on sorting each column of X and that sorting has an n log(n) time complexity,
            it is recommended to use subsampling on datasets with a very large number of samples.
                ***Deprecated since version 1.1: In version 1.3 and onwards, subsample=2e5 will be the default.
        random_state: int, RandomState instance or None, default=None
            Determines random number generation for subsampling. Pass an int for reproducible results across
            multiple function calls.
        """
        est = KBinsDiscretizer(
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            dtype=dtype,
            subsample=subsample,
            random_state=random_state)
        est.fit(X)
        return est.transform(X), est
    
    def LabelBinarizer(self,
            X,
            neg_label=0,
            pos_label=1,
            sparse_output=False):
        """
        neg_label: int, default=0
            Value with which negative labels must be encoded.
        pos_label: int, default=1
            Value with which positive labels must be encoded.
        sparse_output: bool, default=False
            True if the returned array from transform is desired to be in sparse CSR format.
        """
        lb = LabelBinarizer(
            neg_label=neg_label,
            pos_label=pos_label,
            sparse_output=sparse_output)
        lb.fit(X)
        return lb.transform(X), lb

    def LabelEncoder(self, X):
        le = LabelEncoder()
        le.fit(X)
        return le.transform(X), le
    
    def MultiLabelBinarizer(self,
            X,
            classes=None,
            sparse_output=False):
        """
        classesarray-like of shape (n_classes,), default=None
            Indicates an ordering for the class labels. All entries should be unique (cannot contain duplicate classes).
        sparse_outputbool, default=False
            Set to True if output binary array is desired in CSR sparse format.
        """
        mlb = MultiLabelBinarizer(
            classes=classes,
            sparse_output=sparse_output)
        mlb.fit(X)
        return mlb.transform(X), mlb
    
    def OneHotEncoder(self,
            X,
            categories='auto',
            drop=None,
            sparse='deprecated',
            # dtype=<class 'numpy.float64'>,
            handle_unknown='error',
            min_frequency=None,
            max_categories=None):
        """
        categories: 'auto' or a list of array-like, default='auto'
            Categories (unique values) per feature:
                'auto' : Determine categories automatically from the training data.
                list : categories[i] holds the categories expected in the ith column. The passed categories should not
                    mix strings and numeric values within a single feature, and should be sorted in case of numeric values.
                **The used categories can be found in the categories_ attribute.
        drop: {'first', 'if_binary'} or an array-like of shape (n_features,), default=None
            Specifies a methodology to use to drop one of the categories per feature. This is useful in situations where
            perfectly collinear features cause problems, such as when feeding the resulting data into an unregularized
            linear regression model.
            However, dropping one category breaks the symmetry of the original representation and can therefore induce
            a bias in downstream models, for instance for penalized linear classification or regression models.
                None : retain all features (the default).
                'first' : drop the first category in each feature. If only one category is present,
                    the feature will be dropped entirely.
                'if_binary' : drop the first category in each feature with two categories.
                    Features with 1 or more than 2 categories are left intact.
                array : drop[i] is the category in feature X[:, i] that should be dropped.
                *** Changed in version 0.23: The option drop='if_binary' was added in 0.23.
        sparse: bool, default=True
            Will return sparse matrix if set True else will return an array.
            *** Deprecated since version 1.2: sparse is deprecated in 1.2 and will be removed in 1.4.
                Use sparse_output instead.
        dtype: number type, default=float
            Desired dtype of output.
        handle_unknown{'error', 'ignore', 'infrequent_if_exist'}, default='error'
            Specifies the way unknown categories are handled during transform.
                'error' : Raise an error if an unknown category is present during transform.
                'ignore' : When an unknown category is encountered during transform,
                    the resulting one-hot encoded columns for this feature will be all zeros.
                    In the inverse transform, an unknown category will be denoted as None.
                'infrequent_if_exist' : When an unknown category is encountered during transform,
                    the resulting one-hot encoded columns for this feature will map to the infrequent
                    category if it exists. The infrequent category will be mapped to the last position
                    in the encoding. During inverse transform, an unknown category will be mapped to the
                    category denoted 'infrequent' if it exists. If the 'infrequent' category does not exist,
                    then transform and inverse_transform will handle an unknown category as with
                    handle_unknown='ignore'. Infrequent categories exist based on min_frequency and max_categories.
                *** Changed in version 1.1: 'infrequent_if_exist' was added to automatically handle unknown categories
                    and infrequent categories.
        min_frequency: int or float, default=None
            Specifies the minimum frequency below which a category will be considered infrequent.
            If int, categories with a smaller cardinality will be considered infrequent.
            If float, categories with a smaller cardinality than min_frequency * n_samples will be considered infrequent.
        max_categories: int, default=None
            Specifies an upper limit to the number of output features for each input feature when considering
            infrequent categories. If there are infrequent categories, max_categories includes the category
            representing the infrequent categories along with the frequent categories. If None, there is no limit
            to the number of output features.
        """
        enc = OneHotEncoder(
            categories=categories,
            drop=drop,
            sparse=sparse,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            max_categories=max_categories)
        enc.fit(X)
        return enc.transform(X), enc
    
    def OrdinalEncoder(self,
            X,
            categories='auto',
            # dtype=<class 'numpy.float64'>,
            handle_unknown='error',
            unknown_value=None,
            encoded_missing_value=nan):
        """
        categories: 'auto' or a list of array-like, default='auto'
            Categories (unique values) per feature:
                'auto' : Determine categories automatically from the training data.
                list : categories[i] holds the categories expected in the ith column.
                    The passed categories should not mix strings and numeric values,
                    and should be sorted in case of numeric values.
                The used categories can be found in the categories_ attribute.
        dtype: number type, default np.float64
            Desired dtype of output.
        handle_unknown: {'error', 'use_encoded_value'}, default='error'
            When set to 'error' an error will be raised in case an unknown categorical
            feature is present during transform. When set to 'use_encoded_value',
            the encoded value of unknown categories will be set to the value given for
            the parameter unknown_value. In inverse_transform, an unknown category
            will be denoted as None.
        unknown_value: int or np.nan, default=None
            When the parameter handle_unknown is set to 'use_encoded_value',
            this parameter is required and will set the encoded value of unknown categories.
            It has to be distinct from the values used to encode any of the categories in fit.
            If set to np.nan, the dtype parameter must be a float dtype.
        encoded_missing_value: int or np.nan, default=np.nan
            Encoded value of missing categories.
            If set to np.nan, then the dtype parameter must be a float dtype.
        """
        enc = OrdinalEncoder(
            categories=categories,
            handle_unknown=handle_unknown,
            unknown_value=unknown_value,
            encoded_missing_value=encoded_missing_value)
        enc.fit(X)
        return enc.transform(X), enc
