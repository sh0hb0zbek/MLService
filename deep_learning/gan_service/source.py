import numpy
from numpy import zeros
from numpy import ones
from numpy import linspace
from numpy import asarray
from numpy import arccos
from numpy import clip
from numpy import dot
from numpy import sin
from numpy import vstack
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from numpy.random import randn
from numpy.random import randint
from numpy.linalg import norm
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.constraints import Constraint
from skimage.transform import resize
from scipy.linalg import sqrtm
from layers import layer

def main_gan(args):
    """
    argv_discriminator = {
                         'model_type': 'Model',
                         'layers': [layers.layer(...),
                                    layers.layer(...),
                                    layers.layer(...),
                                    layers.layer(...),
                                    ...],
                         'optimizer': layers.optimzer(...),
                         'loss': '...',
                         'metrics': '...'
    }
    
    argv_generator = {
                     'model_type': 'Model',
                     'layers': [layers.layer(...),
                                layers.layer(...),
                                layers.layer(...),
                                layers.layer(...),
                                ...],
                     'optimizer': layers.optimzer(...),
                     'loss': '...',
                     'metrics': '...',
                     'do_compile': False  # --> if model is generator
    }
    
    argv_gan = {
                'optimizer': layers.optimizer(...),
                'loss': ...,
                'model_type': 'Sequential' # or 'Model'
    }
    
    argv_train = {
                  'n_epochs': ...,
                  'n_batch': ...
    }
    
    dataset --> training dataset
    """

    argv_discriminator = args[0]
    argv_generator = args[1]
    argv_gan = args[2]
    argv_train = args[3]
    dataset = args[4]
    
    d_model = define_model(**argv_discriminator)
    g_model = define_model(**argv_generator)
    gan_model = define_gan(g_model, d_model, **argv_gan)
    train(g_model, d_model, gan_model, dataset, **argv_train)

# function for scaling images
def scale_images(images, scale=[-1, 1]):
    # convert from unit8 to float32
    converted_images = images.astype('float32')
    # scale
    min_value = converted_images.max()
    max_value = converted_images.min()
    average_value = (max_value - min_value) / (scale[1] - scale[0])
    converted_images = (images - min_value) / average_value + scale[0]
    return converted_images

# function for generating points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space and reshape into a batch of inputs for the network
    return randn(latent_dim * n_samples).reshape((n_samples, latent_dim))

# define the combined generator and discriminator model, for updating generator
def define_gan(g_model, d_model, optimizer, loss='binary_crossentropy', model_type='Sequential'):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    if model_type == 'Sequential':
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add discriminator
        model.add(d_model)
    elif model_type == 'Model':
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get image output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gan_noise, gen_label], gan_output)
    # compile model
    model.compile(loss=loss, optimizer=optimizer)
    return model

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y

# generate a batch of images, returns images and targets
def generate_fake_samples_2(g_model, dataset, path_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# select real samples from dataset
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return x, y

# select a batch of random samples, returns images and target
def generate_real_samples_2(dataset, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (0)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# uniform interpolate between two points in latent space
def interpolate_points(p1, p2, intrpl, n_steps=10):
    # interpolate ratios betweem the points
    ratios = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in rations:
        v = intrpl(ratio, p1, p2)
        vectors.append(v)
    return asarray(vectors)

# spherical linear interpolation (slerp)
def slerp(val, low, high):
    omega = arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))
    so = sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val)*low + val*high
    return sin((1.0-val)*omega) / so*low + sin(val*omega) / so * high

# uniform interpolation (uni_inter)
def uni_inter(val, low, high):
    return (1.0-val)*p1 + val*p2

# average list of latent space vectors
def average_points(points, ix):
    # convert to zero offset points
    zero_ix = [i-1 for i in ix]
    # retreive required points
    vectors = points[zero_ix]
    # average the vectors
    avg_vector = mean(vectors, axis=0)
    # combine original and avg vectors
    all_vectors = vstack((vectors, avg_vector))
    return all_vectors

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1e-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx*(log(p_yx+eps)-log(p_y+eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbot interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# assumes images have any shape and pixels in [0, 255]
def calculate_inception_score(images, n_split=10, eps=1e-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0]/n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i*n_part, (i+1)*n_part
        subset = images[ix_start:ix_end]
        # convert from unit8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299, 299, 3))
        # pre-process images, scale [-1, 1]
        sunset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dim(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx*(log(p_yx+eps)-log(p_y+eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

# calculate frechet inception distance (FID) using NumPy
def calculate_fid_w_numpy(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# calculate frechet inception distance (FID) using inception v3 model (keras)
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # use calculate_fid_w_numpy() to calculate FID
    return calculate_fid_w_numpy(act1, act2)

# clip model weights to a given hypercube (constraint)
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# define a model
def define_model(lyrs, model_type='Sequential', optimizer='rmsprop', loss=None, metrics=None, do_compile=True):
    if model_type == 'Sequential':
        model = Sequential()
        for lyr in lyrs:
            model.add(lyr)
    elif model_type == 'Model':
        if 'Concatenate' in lyrs:
            idx = lyrs.index('Concatenate')
            idx_end = lyrs.index('End')
            in_label = lyrs[0]
            li = in_label
            for i in range(1, idx):
                li = lyrs[i](li)
            in_image = lyrs[idx+1]
            lii = in_image
            for i in range(idx+2, idx_end):
                lii = lyrs[i](lii)
            merge = layer('Concatenate')([lii, li])
            fe = merge
            for i in range(idx_end+1, len(lyrs)):
                fe = fe(lyrs[i])
            inputs = [in_image, in_label]
            outputs = fe
        else:
            in_image = layers[0]
            li = in_image
            for i in range(1, len(layers)):
                li = layers[i](li)
            inputs = in_image
            outputs = li
        model = Model(inputs=inputs, outputs=outputs)
    if do_compile:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# train the generator and discriminator models
def train(g_model, d_model, gan_model, dataset, latent_dim,
         n_epochs=10, n_batch=128, do_print=False, save_model=True):
    # calculate the number of batches per epoch
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    # calculate the total itereations based on batch and epoch
    n_steps = batch_per_epoch * n_epochs
    # calculate the number of samples in half a batch
    half_batch = int(n_batch/2)
    # prepare lists for storing stats each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        epoch_n =int(i/batch_per_epoch)
        # get randomly selected "real" samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update dicriminator model weights
        d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
        # genereate "fake" samples
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))
        # update the generator via discriminator's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        a1_hist.append(d_acc1)
        a2_hist.append(d_acc2)
        g_hist.append(g_loss)
        if do_print and (i+1)%batch_per_epoch == 0:
            print(f">{epoch_n+1}/{n_epochs}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}, a1={100*d_acc1:.1f}%, a2={100*d_acc2:.1f}%")
        if save_model:
            if(i+1)%batch_per_epoch == 0 and (epoch_n+1)%5 == 0:
                g_model.save(f"generator_{epoch_n+1:03d}.h5")

def train_main_gan():
    import layers
    from source import main_gan
    from keras.datasets.mnist import load_data
    from numpy import expand_dims

    init = layers.initializer('RandomNormal', stddev=0.02)
    input_shape = (28, 28, 1)
    optimizer = layers.optimizer('Adam', learning_rate=0.0002, beta_1=0.5)
    n_nodes = 128 * 7 * 7
    latent_dim = 50

    argv_discriminator = {
        'model_type': 'Sequential',
        'lyrs': [
            # downsample to 14x14
            layers.layer('Conv2D', filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init, input_shape=input_shape),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # downsample to 7x7
            layers.layer('Conv2D', filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # classifier
            layers.layer('Flatten'),
            layers.layer('Dense', units=1, activation='sigmoid')
        ],
        'optimizer': optimizer,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    }
    argv_generator = {
        'model_type': 'Sequential',
        'lyrs': [
            # foundation for 7x7 image
            layers.layer('Dense', units=n_nodes, kernel_initializer=init, input_dim=latent_dim),
            layers.layer('LeakyReLU', alpha=0.2),
            layers.layer('Reshape', target_shape=(7, 7, 128)),
            # upsample to 14x14
            layers.layer('Conv2DTranspose', filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # upsample to 28x28
            layers.layer('Conv2DTranspose', filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # output 28x28x1
            layers.layer('Conv2D', filters=1, kernel_size=(7, 7), activation='tanh', padding='same', kernel_initializer=init)
        ],
        'do_compile': False
    }
    argv_gan = {
        'optimizer': optimizer,
        'loss': 'binary_crossentropy',
        'model_type': 'Sequential'
    }
    argv_train = {'latent_dim': latent_dim}

    # load data
    (train_x, train_y), (_, _) = load_data()
    # expand to 3d
    x = expand_dims(train_x, axis=-1)
    selected_ix = train_y == 8
    x = x[selected_ix]
    # convert from ints to floats and scale from [0, 255] to [-1, 1]
    x = x.astype('float32')
    dataset = (x - 127.5) / 127.5

    argv = [
        argv_discriminator,
        argv_generator,
        argv_gan,
        argv_train,
        dataset
    ]
    main_gan(argv)
