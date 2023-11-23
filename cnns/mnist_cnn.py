from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


# Load train and test data
# ((train_images, train_labels), (val_images, val_labels)), ds_name = mnist.load_data(), 'digits'  # 0.99
((train_images, train_labels), (val_images, val_labels)), ds_name = fashion_mnist.load_data(), 'fashion'  # 0.91

# Plot 100 training images
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
axs = ax.ravel()
for i in range(100):
    axs[i].imshow(train_images[i], cmap='Greys')
    axs[i].axis('off')

plt.tight_layout()
plt.savefig('../data/plots/mnist_{}_100.jpg'.format(ds_name))
plt.close()

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

# Define hyperparameters in dictionary for flexible use
config = {'conv_layers': 1,
          'conv_filters': 8,
          'conv_kernel_size': 3,
          'conv_initializer': 'he_uniform',
          'conv_padding': 'same',
          'conv_activation_function': 'relu',
          'conv_dropout_rate': 0,
          'maxpool_stride': 2,
          'maxpool_kernel_size': 2,
          'fc_layers': 2,
          'fc_neurons': 16,
          'fc_activation_function': 'relu',
          'fc_initializer': 'he_uniform',
          'fc_dropout_rate': 0,
          'learning_rate': 0.001,
          'momentum': 0.9,
          'loss': 'categorical_crossentropy',
          'epochs': 2}

# Define model architecture
model = Sequential()

# First convolutional and pooling layer
model.add(Conv2D(input_shape=(28, 28, 1), filters=config['conv_filters'], kernel_size=config['conv_kernel_size'], padding=config['conv_padding'], activation=config['conv_activation_function'], kernel_initializer=config['conv_initializer']))
model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))

# Convolutional layers
for i in range(config['conv_layers']):
    model.add(Conv2D(filters=config['conv_filters'], kernel_size=config['conv_kernel_size'], padding=config['conv_padding'], activation=config['conv_activation_function'], kernel_initializer=config['conv_initializer']))
    model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))

    if config['conv_dropout_rate'] > 0.0:
        model.add(Dropout(config['conv_dropout_rate']))

# Global average pooling reduces number of dimensions
model.add(GlobalAveragePooling2D())

# Dense layers
for i in range(config['fc_layers']):
    model.add(Dense(units=config['fc_neurons'], activation=config['fc_activation_function'], kernel_initializer=config['fc_initializer']))

    if config['fc_dropout_rate'] > 0.0:
        model.add(Dropout(config['fc_dropout_rate']))

# Add last dense layer with neurons = number of classes
model.add(Dense(units=nclasses, activation='softmax', kernel_initializer=config['fc_initializer']))

# Compile model
model.compile(optimizer=SGD(lr=config['learning_rate'], momentum=config['momentum']), loss=config['loss'], metrics=['accuracy'])

# Print model architecture
model.summary()

# Tensorboard callback
tensorboard_callback = TensorBoard(log_dir='tensorboard/mnist_{}_cnn_{}'.format(ds_name, datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')))

# Train model
model.fit(x=train_images, y=train_labels, epochs=config['epochs'], validation_data=(val_images, val_labels), callbacks=[tensorboard_callback])

# Evaluate model
val_loss, val_acc = model.evaluate(val_images, val_labels)
print('MNIST {} model - val. accuracy: {:.2f}'.format(ds_name, val_acc))
