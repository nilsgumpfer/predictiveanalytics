from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import normalize_heatmap
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def prepare_data():
    # Load train and test data
    ((train_images, train_labels), (val_images, val_labels)), ds_name = fashion_mnist.load_data(), 'fashion'

    # Normalize color values (here: grey-scales)
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Expand pixel dimension (1 color channel)
    train_images = np.expand_dims(train_images, axis=3)
    val_images = np.expand_dims(val_images, axis=3)

    # Do one-hot encoding / do categorical conversion
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    # Extract number of classes from data dimensions
    nclasses = np.shape(train_labels)[1]

    return train_images, train_labels, val_images, val_labels, nclasses, ds_name


def train_model(config, train_images, train_labels, val_images, val_labels, nclasses, ds_name):
    # Define model architecture
    model = Sequential()

    # First convolutional and pooling layer
    model.add(Conv2D(input_shape=(28, 28, 1), filters=config['conv_filters'], kernel_size=config['conv_kernel_size'],
                     padding=config['conv_padding'], activation=config['conv_activation_function'],
                     kernel_initializer=config['conv_initializer']))
    model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))

    # Convolutional layers
    for i in range(config['conv_layers']):
        model.add(Conv2D(filters=config['conv_filters'], kernel_size=config['conv_kernel_size'],
                         padding=config['conv_padding'], activation=config['conv_activation_function'],
                         kernel_initializer=config['conv_initializer']))
        model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))

        if config['conv_dropout_rate'] > 0.0:
            model.add(Dropout(config['conv_dropout_rate']))

    # Global average pooling reduces number of dimensions
    model.add(GlobalAveragePooling2D())

    # Dense layers
    for i in range(config['fc_layers']):
        model.add(Dense(units=config['fc_neurons'], activation=config['fc_activation_function'],
                        kernel_initializer=config['fc_initializer']))

        if config['fc_dropout_rate'] > 0.0:
            model.add(Dropout(config['fc_dropout_rate']))

    # Add last dense layer with neurons = number of classes
    model.add(Dense(units=nclasses, activation='softmax', kernel_initializer=config['fc_initializer']))

    # Compile model
    model.compile(optimizer=SGD(lr=config['learning_rate'], momentum=config['momentum']), loss=config['loss'],
                  metrics=['accuracy'])

    # Print model architecture
    model.summary()

    # Tensorboard callback
    tensorboard_callback = TensorBoard(log_dir='./tensorboard/mnist_{}_cnn_{}'.format(ds_name, datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')), histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(filepath='./models/mnist_{}_cnn.h5'.format(ds_name), monitor='val_accuracy', mode='max', save_best_only=True)

    # Train model
    model.fit(x=train_images, y=train_labels, epochs=config['epochs'], validation_data=(val_images, val_labels),
              callbacks=[tensorboard_callback, checkpoint_callback])

    # Evaluate model
    val_loss, val_acc = model.evaluate(val_images, val_labels)
    print('MNIST {} model - val. accuracy: {:.2f}'.format(ds_name, val_acc))

    return model


def main(config, train=False):
    # Load data
    train_images, train_labels, val_images, val_labels, nclasses, ds_name = prepare_data()

    if train:
        # Train model
        model = train_model(config, train_images, train_labels, val_images, val_labels, nclasses, ds_name)
    else:
        # Load model
        model = load_model('./models/mnist_fashion_cnn.h5')

    # Remove softmax
    model.layers[-1].activation = None

    # Calculate relevancemaps
    i = np.random.randint(low=0, high=len(val_images))
    x = val_images[i]
    R1 = calculate_relevancemap('smoothgrad_x_sign', np.array(x), model, mu=0.5)
    R2 = calculate_relevancemap('grad_cam', np.array(x), model, last_conv_layer_name='conv2d_1')
    R1_n = normalize_heatmap(R1)
    R2_n = normalize_heatmap(R2)
    R1_n[R1_n < 0] = 0
    R2_n[R2_n < 0] = 0

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
    axs[0].imshow(x, cmap='gist_gray_r', clim=(-1, 1))
    axs[0].set_title('input')
    axs[1].matshow(R1_n, cmap='seismic', clim=(-1, 1))
    axs[1].set_title('SmoothGrad x SIGN')
    axs[2].matshow(R2_n, cmap='seismic', clim=(-1, 1))
    axs[2].set_title('Grad CAM')

    plt.show()


if __name__ == '__main__':
    # Define hyperparameters
    params = {'conv_layers': 3,
              'conv_filters': 32,
              'conv_kernel_size': 3,
              'conv_initializer': 'he_uniform',
              'conv_padding': 'same',
              'conv_activation_function': 'relu',
              'conv_dropout_rate': 0.1,
              'maxpool_stride': 2,
              'maxpool_kernel_size': 2,
              'fc_layers': 2,
              'fc_neurons': 32,
              'fc_activation_function': 'relu',
              'fc_initializer': 'he_uniform',
              'fc_dropout_rate': 0.25,
              'learning_rate': 0.001,
              'momentum': 0.0,
              'loss': 'categorical_crossentropy',
              'epochs': 50}

    while(True):
        main(params, train=False)

# test