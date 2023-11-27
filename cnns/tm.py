import numpy as np
from PIL import ImageOps
from PIL.Image import Image
from keras import Sequential, Model
from keras.models import load_model
from keras.layers import deserialize
from keras.src.engine.input_layer import InputLayer
from keras.src.layers import Add
from keras.src.models import clone_model
from keras.src.utils import plot_model
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

from cnns.grad_cam import make_gradcam_heatmap, save_and_display_gradcam

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./models/keras_model.h5", compile=False)

# Load the labels
class_names = open("./models/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Image path
img_path = "../data/horse.jpeg"

# Replace this with the path to your image
image = Image.open(img_path).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)

# Convert model
conv_layers = model.layers[0].layers[0].layers
gap_layer = model.layers[0].layers[1]
dense_layers = model.layers[1].layers
layers = [gap_layer] + dense_layers

model2 = clone_model(model.layers[0].layers[0])
x = model2.output

for lyr in layers:
   lyr_cls = str(type(lyr)).rsplit('.', maxsplit=1)[1][:-2]

   cnf = lyr.get_config()
   layer = deserialize({'class_name': lyr_cls, 'config': cnf})

   x = layer(x)

model3 = Model(inputs=model2.inputs, outputs=x, name="MyModel")
for i, l in enumerate(conv_layers):
   model3.layers[i].set_weights(conv_layers[i].get_weights())

model3.layers[-3].set_weights(gap_layer.get_weights())
model3.layers[-2].set_weights(dense_layers[0].get_weights())
model3.layers[-1].set_weights(dense_layers[1].get_weights())

# Predicts the model
prediction = model3.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

# Remove last layer's softmax activation (for the explanation we need the raw values!)
model.layers[1].layers[-1].activation = None

# Generate class activation heatmap (CAM)
heatmap = make_gradcam_heatmap(data, model, model.layers[0].layers[0].layers[-1], idx=None)

# Display heatmap
# plt.matshow(heatmap, cmap=cmap_name)
# plt.show()

# Create and save superimposed visualization
save_and_display_gradcam(img_path, heatmap, '../data/plots/horse_cam.jpg', 'gist_heat', alpha=2.0)