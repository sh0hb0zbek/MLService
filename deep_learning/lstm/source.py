import tensorflow
from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
import inspect


def model_define(model_type, layers):
    """
    model_type: <str>
        model type for neural network. "Sequential" or "model"
    layers: <list>
        layer types and its arguments as <dict> type
        e.g
        layers=[
            {"LSTM": {
                "units": 64,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "use_bias": True,
                ...
            }},
            {"Dense": {
                "units": 10,
                "activation": "relu",
                "kerbel_initializer": "glorot_uniform",
                ...
            }}
        ]
    """
    if model_type == "Sequential":
        model = Sequential()
        for layer in layers:
            model.add(eval(layer[0])(**layer[1]))
    return model

def get_args(signiture, dont_get=list(), lst=list()):
    args = dict()
    for k, v in signiture.parameters.items():
        if k in dont_get or k == 'kwargs':
            continue
        args[k] = v.default

    for key, value in args.items():
        if value is inspect.Parameter.empty: value = "non-exist"
        new_value = input(f"\t{key} [defaul: {value}]: ")
        if new_value == "": new_value=value
        type_value = type(value)
        if key in lst:
            args[key] = [new_value]
        else:
            if type_value is not type(None):
                args[key] = type_value(new_value)
            else:
                args[key] = new_value
    return args


def model_build():
    # make a model
    model_type = input("Type of model: ") # "Sequential" or "Model"
    layers = list()
    ### in this part layer information can be done by sending request with layers arguments and default values
    ### user should input values for each given argument and in the end all layers with argument will be send back
    while True:
        if input("Add a new layer? (y/n)") == "n": break
        layer_type = input("Name of layer: ")
        layers.append([layer_type, get_args(inspect.signature(eval(layer_type)))])
    ### line 66-69 is alternative option to show how layer adding works in terminal
    model = model_define(model_type=model_type, layers=layers)
    # compile
    optimizer = eval(input("Optimizer: "))
    opt = optimizer(**get_args(inspect.signature(optimizer)))
    print("Compile the model: ")
    model.compile(optimizer=opt, **get_args(inspect.signature(model.compile), ['optimizer'], ['metrics']))
    return model
