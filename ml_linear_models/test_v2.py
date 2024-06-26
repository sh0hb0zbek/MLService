from source_v2 import *
import sys

def linear_model_test(
    model_name,
    dataset_filepath,
    sort=False,
    scaler_type=None,
    save=False,
    filename=None
):
    # loading
    """
    required arguments for loading data
        - dataset_filepath
        - sort (default=False)
    """
    # filetype = data_filetype(dataset_filepath)
    # dataset = datafile_loader(dataset_filepath, filetype=filetype)
    # X, y = data_sorting(dataset, sort)
    X, y = data_sorting(
        dataset=datafile_loader(filepath=dataset_filepath, filetype=data_filetype(filepath=dataset_filepath)),
        sort=sort)

    # pre-processing
    """
    required arguments for pre-processing
        - scaler_type (default=None)
        - arguments of train_test_split method, e.g:
            - test_size
            - train_size
            - random_state
            - shuffle
    """
    test_size = 0.3
    random_state = 15
    x_train, x_test, y_train, y_test = data_split(X, y, scaler_type=scaler_type, test_size=test_size, random_state=random_state)

    # fitting
    """
    required arguments for model fitting
        - model_name
    """
    model = LinearModel(model_name=model_name)

    if model.model_type == 'classifier':
        y_train, y_test = scaler('LabelEncoder', y_train, y_test)
    model.fit(x_train=x_train, y_train=y_train)

    # evaluating and visualizing
    model.plot(x_test, y_test)

    # saving
    """
    required argument for model saving
        - save (default=False)
        - filename (default=None)
    """
    if save:
        model.save(filename=filename)


if __name__ == '__main__':
    args = argparse_linear_models()
    linear_model_test(
        model_name=args.model_name,
        dataset_filepath=args.filepath,
        sort=args.sort,
        scaler_type=args.scaler_type,
        save=args.save,
        filename=args.filename
    )
