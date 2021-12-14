from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical

# Load train and test data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Plot 100 training samples
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
axs = ax.ravel()
for i in range(100):
    axs[i].imshow(train_images[i], cmap='Greys')

plt.tight_layout()
plt.savefig('../data/mnist_fashion_100.jpg')
plt.close()

# Normalize color values (here: grey-scales)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Do one-hot encoding / do categorical conversion
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Extract number of classes from data dimensions
classes = np.shape(train_labels)[1]

# Define model architecture
model = Sequential()

# Flatten input to one dimension
model.add(Flatten(input_shape=(28, 28)))

# Add several fully-connected layers
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))

# Last dense layer has width of label (number of classes)
model.add(Dense(units=classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model architecture
model.summary()

# Tensorboard callback
tensorboard_callback = TensorBoard(log_dir='tensorboard/' + 'mnist_fashion_fc_' + datetime.utcnow().strftime('%Y%m%d%H%M%S'))

# Train model
model.fit(x=train_images, y=train_labels, epochs=20, validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)