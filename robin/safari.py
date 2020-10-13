import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import os, shutil
from keras.applications.vgg16 import preprocess_input, decode_predictions

conv_base = VGG16(weights='imagenet',
                  include_top=True,
                  input_shape=(224, 224, 3))

def preprocess_image(img_path,target_size=(224, 224)):
    # load image (make sure it is 150x150)
    img = image.load_img(img_path, target_size=target_size)

    # convert to rgb array
    x = image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 150, 150, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)

    return x

base_dir = '/Users/robinvanderveken/Projects/Safari'
test_image = preprocess_image(os.path.join(base_dir,'rhino','001.jpg'),(150,150))

pred = conv_base.predict(test_image)
decode_predictions(pred,top=5)