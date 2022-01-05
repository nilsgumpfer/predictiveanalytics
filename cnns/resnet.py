from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

# https://arxiv.org/abs/1512.03385
model_rn50 = ResNet50(weights='imagenet')
model_rn101 = ResNet101(weights='imagenet')
model_rn152 = ResNet152(weights='imagenet')

img_path = '../data/elephant.jpg'
# img_path = '../data/rooster.jpg'
# img_path = '../data/myrooster.jpg'
# img_path = '../data/castlebicycle.jpg'
# img_path = '../data/castlebicycle_castle.jpg'
# img_path = '../data/castlebicycle_bike.jpg'
# img_path = '../data/castle.jpg'
# img_path = '../data/castledark.jpg'
# img_path = '../data/giraffe.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

pred_rn50 = model_rn50.predict(x)
print('Top 3 predictions (ResNet50):', decode_predictions(pred_rn50, top=3)[0])

pred_rn101 = model_rn101.predict(x)
print('Top 3 predictions (ResNet101):', decode_predictions(pred_rn101, top=3)[0])

pred_rn152 = model_rn152.predict(x)
print('Top 3 predictions (ResNet152):', decode_predictions(pred_rn152, top=3)[0])
