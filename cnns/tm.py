from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from matplotlib import pyplot as plt
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import (load_image, aggregate_and_normalize_relevancemap_rgb, download_image, calculate_explanation_innvestigate)


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

# Replace this with the path to your image
image = Image.open("../data/horse.jpeg").convert("RGB")
# image = Image.open("../data/horse_watermark.jpg").convert("RGB")

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
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

# Remove last layer's softmax activation (for the explanation we need the raw values!)
model.layers[-1].activation = None

# TODO: adjust layers

# Calculate relevancemaps
R1 = calculate_relevancemap('lrpz_epsilon_0_5_std_x', np.array(data[0]), model)
R2 = calculate_relevancemap('lrpsign_epsilon_0_5_std_x', np.array(data[0]), model)

print(np.sum(np.ravel(R1-R2)))

# Visualize heatmaps
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 12))
axs[0].imshow(image_array)
axs[1].matshow(aggregate_and_normalize_relevancemap_rgb(R1), cmap='seismic', clim=(-1, 1))
axs[2].matshow(aggregate_and_normalize_relevancemap_rgb(R2), cmap='seismic', clim=(-1, 1))

plt.show()