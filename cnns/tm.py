import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from PIL.Image import Resampling
from keras import Model
from keras.layers import deserialize
from keras.models import load_model, clone_model
from tensorflow import keras


def main(img_path, explain_class_idx=None):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("models/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("./models/labels.txt", "r").read().split('\n')[:-1]

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Image path
    # img_path = '/home/nils/Downloads/dogs/18-8ccab938b95c780e0090a0b533205477.jpg'

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

    # Model prediction
    prediction = model3.predict(data)
    idx_pred = np.argmax(prediction)
    class_name = class_names[idx_pred]
    confidence_score = prediction[0][idx_pred]

    if explain_class_idx is not None:
        idx_to_be_explained = explain_class_idx
    else:
        idx_to_be_explained = idx_pred

    # Print prediction and confidence score
    print("\nPredictions for: ", extract_filename(img_path))
    for i in range(len(class_names)):
        print("Class:", class_names[i][2:], end="")
        print("Confidence Score:", prediction[0][i])

    # Remove last layer's softmax activation (for the explanation we need the raw values!)
    model3.layers[-1].activation = None
    # model3.summary()

    # Generate class activation heatmap (CAM)
    heatmap = make_gradcam_heatmap(data, model3, 'out_relu', idx=idx_to_be_explained)

    # Create and save superimposed visualization
    save_and_display_gradcam(img_path, heatmap, '../data/plots/{}_{}.jpg'.format(extract_filename(img_path), class_names[idx_to_be_explained]), 'jet', alpha=2.0)


def extract_filename(path):
    return str(path).rsplit('/', maxsplit=1)[1].rsplit('.', maxsplit=1)[0]


def make_gradcam_heatmap(img_array, model, last_conv_layer=None, idx=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if idx is None:
            pred_index = tf.argmax(preds[0])
        else:
            pred_index = idx

        print('Explaining idx={}'.format(pred_index))

        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by 'how important this channel is' with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path, cmap_name, alpha):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use colormap to colorize heatmap
    cmap = cm.get_cmap(cmap_name)

    # Use RGB values of the colormap
    cmap_colors = cmap(np.arange(256))[:, :3]
    cmap_heatmap = cmap_colors[heatmap]

    # Create an image with RGB colorized heatmap
    cmap_heatmap = keras.preprocessing.image.array_to_img(cmap_heatmap)
    cmap_heatmap = cmap_heatmap.resize((img.shape[1], img.shape[0]))
    cmap_heatmap = keras.preprocessing.image.img_to_array(cmap_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = cmap_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


# main(img_path='../data/hyaene.jpg', explain_class_idx=0)
# main(img_path='../data/hyaene.jpg', explain_class_idx=1)
# main(img_path='../data/hyaene2.jpg', explain_class_idx=0)
# main(img_path='../data/hyaene2.jpg', explain_class_idx=1)
main(img_path='../data/hyaene3.png', explain_class_idx=0)
main(img_path='../data/hyaene3.png', explain_class_idx=1)
