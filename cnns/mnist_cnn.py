from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.datasets import mnist, fashion_mnist
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model


# Load train and test data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # 0.9904999732971191
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # 0.910099983215332
# TODO: check overfitting capability

# Plot 100 training images
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
axs = ax.ravel()
for i in range(100):
    axs[i].imshow(train_images[i], cmap='Greys')
    axs[i].axis('off')

plt.tight_layout()
plt.savefig('../data/mnist_100.jpg')
plt.close()

# Normalize color values (here: grey-scales)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Expand pixel dimension (1 color channel)
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Do one-hot encoding / do categorical conversion
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Extract number of classes from data dimensions
nclasses = np.shape(train_labels)[1]

# Define hyperparameters in dictionary for flexible use
config = {'conv_layers': 2,
          'conv_filters': 64,
          'conv_kernel_size': 3,
          'conv_initializer': 'he_uniform',
          'conv_padding': 'same',
          'conv_activation_function': 'relu',
          'conv_dropout_rate': 0.1,
          'maxpool_stride': 2,
          'maxpool_kernel_size': 2,
          'fc_layers': 2,
          'fc_neurons': 100,
          'fc_activation_function': 'relu',
          'fc_initializer': 'he_uniform',
          'fc_dropout_rate': 0.1,
          'learning_rate': 0.01,
          'momentum': 0.9,
          'loss': 'categorical_crossentropy',
          'epochs': 10}

# Define model architecture
model = Sequential()

# First convolutional and pooling layer
model.add(Conv2D(input_shape=(28, 28, 1), filters=config['conv_filters'], kernel_size=config['conv_kernel_size'], padding=config['conv_padding'], activation=config['conv_activation_function'], kernel_initializer=config['conv_initializer']))
model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))

# Convolutional and dropout layers
for i in range(config['conv_layers']):
    model.add(Conv2D(filters=config['conv_filters'], kernel_size=config['conv_kernel_size'], padding=config['conv_padding'], activation=config['conv_activation_function'], kernel_initializer=config['conv_initializer']))
    model.add(MaxPool2D(strides=config['maxpool_stride'], pool_size=config['maxpool_kernel_size']))
    # model.add(Dropout(config['conv_dropout_rate']))

# Global average pooling reduces number of dimensions
model.add(GlobalAveragePooling2D())

# Dense and dropout layers
for i in range(config['fc_layers']):
    model.add(Dense(units=config['fc_neurons'], activation=config['fc_activation_function'], kernel_initializer=config['fc_initializer']))
    # model.add(Dropout(config['fc_dropout_rate']))

# Add last dense layer with neurons = number of classes
model.add(Dense(units=nclasses, activation='softmax', kernel_initializer=config['fc_initializer']))

# Compile model
model.compile(optimizer=SGD(lr=config['learning_rate'], momentum=config['momentum']), loss=config['loss'], metrics=['accuracy'])

# Print model architecture
model.summary()

# Visualize model
plot_model(model=model,
           to_file='../data/mnist_cnn.jpg',
           show_shapes=True,
           show_layer_names=False,
           expand_nested=False,
           dpi=100)

# Tensorboard callback
tensorboard_callback = TensorBoard(log_dir='tensorboard/' + 'mnist_cnn_' + datetime.utcnow().strftime('%Y%m%d%H%M%S'))

# Train model
model.fit(x=train_images, y=train_labels, epochs=config['epochs'], validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
