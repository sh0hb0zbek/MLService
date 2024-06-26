from linear_models import linear_model_main
import argparse

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',   type=str,   required=True,                help='Name of Sci-Kit Linear Models.')
    parser.add_argument('--filepath',     type=str,   required=True,                help='Path for CSV dataset file.')
    parser.add_argument('--test_size',    type=float, required=False, default=0.3,  help='The propotion of the dataset to include in the test.')
    parser.add_argument('--train_size',   type=float, required=False, default=None, help='The propotion of the dataset to include in the train.')
    parser.add_argument('--random_state', type=int,   required=False, default=None, help='Controls the shuffling applied to the data before applying the split.')
    parser.add_argument('--shuffle',      action='store_true',                      help='Whether or not to shuffle the data before splitting.')
    parser.add_argument('--encode',       action='store_true',                      help='Use if there are string features. If True, string features will be encoded by OneHotEncoder, then main function returns fitted model and encoder if target feature is encoded.')
    parser.add_argument('--scaler_type',  type=str,   required=False, default=None, help='Name of scaler of Sci-Kit Preprocessing.')
    parser.add_argument('--eval_type',    type=str,   required=False, default=None, help='Name of evaluation methods of Sci-Kit Metrics.')
    parser.add_argument('--plot_type',    type=str,   required=False, default=None, help='"confusion_matrix" for classification, "True" for regression.')
    opt = parser.parse_args()

    model = linear_model_main(
        model_type=opt.model_type,
        filepath=opt.filepath,
        test_size=opt.test_size,
        train_size=opt.train_size,
        random_state=opt.random_state,
        shuffle=opt.shuffle,
        scaler_type=opt.scaler_type,
        eval_type=opt.eval_type,
        plot_type=opt.plot_type)
