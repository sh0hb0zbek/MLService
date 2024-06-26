from preprocessing import *
from source import *

def clustering_model_test(
        model_name,
        n_clusters,
        dataset_filepath,
        scaler_type=None,
        encoder_type=None,
        save=False,
        filename=None):
    # loading dataset
    """
    required arguments for loading data
        - dataset_filepath
        - sort (default=False)
    """
    X, labels = dataset(data=datafile_loader(filepath=dataset_filepath, filetype=data_filetype(filepath=dataset_filepath)))

    # pre_processing
    if encoder_type is not None:
        encoder = eval("Encoder." + encoder_type)
        X = encoder(X)
    
    # model fitting and plotting
    model = ClusterModel(model_name=model_name)
    model.fit(X, n_clusters)
    model.plot(X, x_label=labels[0], y_label=labels[1])

    if save:
        model.save(filename=filename)


if __name__ == "__main__":
    args = argparse_cluster_models()
    clustering_model_test(
        model_name=args.model_name,
        n_clusters=args.n_clusters,
        dataset_filepath=args.filepath,
        scaler_type=args.scaler_type,
        encoder_type=args.encoder_type,
        save=args.save,
        filename=args.filename)