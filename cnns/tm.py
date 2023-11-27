import numpy as np
from keras import Sequential
from keras.models import load_model
from keras.layers import deserialize

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./models/keras_model.h5", compile=False)

# Load the labels
# class_names = open("./models/labels.txt", "r").readlines()
#
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#
# # Image path
# img_path = "../data/horse.jpeg"
#
# # Replace this with the path to your image
# image = Image.open(img_path).convert("RGB")
#
# # resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#
# # turn the image into a numpy array
# image_array = np.asarray(image)
#
# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
#
# # Load the image into the array
# data[0] = normalized_image_array
#
# # Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]
#
# # Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)

conv_layers = model.layers[0].layers[0].layers
gap_layer = model.layers[0].layers[1]
dense_layers = model.layers[1].layers

layers = conv_layers + [gap_layer] + dense_layers

# config = {'name': 'MyModel', 'build_input_shape': (None, 224, 224, 3), 'layers': []}
# print(config)

model2 = Sequential()

for i, lyr in enumerate(layers):
   if i > 0:
      cnf = lyr.get_config()
      if i == 1:
         cnf['input_shape'] = (224, 224, 3)
      layer = deserialize({'class_name': str(type(lyr)).rsplit('.', maxsplit=1)[1][:-2], 'config': cnf})
      model2.add(layer)

model2.summary()


# print(config['layers'])

# model2 = Sequential().from_config(config)

# model2.summary()




# # Remove last layer's softmax activation (for the explanation we need the raw values!)
# model.layers[1].layers[-1].activation = None
#
# # Generate class activation heatmap (CAM)
# heatmap = make_gradcam_heatmap(data, model, model.layers[0].layers[0].layers[-1], idx=None)
#
# # Display heatmap
# # plt.matshow(heatmap, cmap=cmap_name)
# # plt.show()
#
# # Create and save superimposed visualization
# save_and_display_gradcam(img_path, heatmap, '../data/plots/horse_cam.jpg', 'gist_heat', alpha=2.0)