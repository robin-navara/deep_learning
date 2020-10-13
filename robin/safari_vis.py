import matplotlib.pyplot as plt
from tensorflow import keras
import os, shutil
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import cv2

def image_heatmap(model, x, img_path_out, animals, conv_layer_name = 'conv2d_4'):

    # We add a dimension to transform our array into a "batch"
    # of size (1, 150, 150, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)

    for animal_index in range(len(animals)):
        class_pred = animal_index
        max_category_output = model.output[:,class_pred]

        # latest conv layer is conv2d_8
        last_conv_layer_safari = model.get_layer(conv_layer_name)
        # get gradients of this layer
        grads_safari = K.gradients(max_category_output, last_conv_layer_safari.output)[0]

        pooled_grads_safari = K.mean(grads_safari, axis=(0, 1, 2))

        # This function allows us to access the values of the quantities we just defined:
        # `pooled_grads_safari` and the output feature map of `block5_conv3`,
        # given a sample image
        iterate_safari = K.function([model.input], [pooled_grads_safari, last_conv_layer_safari.output[0]])

        # These are the values of these two quantities, as Numpy arrays,
        # given our sample image of two elephants
        pooled_grads_safari_value, conv_layer_output_value_safari = iterate_safari([x])

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the elephant class
        for i in range(128):
            conv_layer_output_value_safari[:, :, i] *= pooled_grads_safari_value[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(conv_layer_output_value_safari, axis=-1)

        # normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # We use cv2 to load the original image
        img = cv2.imread(img_path)

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img

        write_img_path_out = os.path.join(img_path_out["dir"],img_path_out["file"]+"_"+animals[class_pred])
        cv2.imwrite(write_img_path_out,superimposed_img)

# base dir
model_dir = '/Users/robinvanderveken/Projects/Safari'
base_dir = '/Users/robinvanderveken/Projects/Safari/test'
base_dir_out = '/Users/robinvanderveken/Projects/Safari/test_heatmaps'
# os.mkdir(base_dir_out)
# load model
model_safari = keras.models.load_model(os.path.join(model_dir,'safariv2.h5'))

# create folders
animals = ['buffalo', 'elephant', 'rhino', 'zebra']
for animal in animals:
    os.mkdir(os.path.join(base_dir_out,animal))

def generate_heatmaps(animals):
    for animal in animals:
        figlist = [name for name in os.listdir(os.path.join(base_dir, animal)) if name.split('.')[1] == 'jpg']
        for fig in figlist[1:5]:
            img_path = os.path.join(base_dir,animal,fig)
            img_path_out = {"dir":os.path.join(base_dir_out,"All"),"file":fig}
            image_heatmap(model_safari, img_path, img_path_out, animals)

animal = 'elephant'
figlist
for fig in figlist:
    img_path = os.path.join(base_dir, animal, fig)
    img = image.load_img(img_path, target_size=(224, 224))

    # convert to rgb array
    x = image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 150, 150, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)

    # normalize values
    x /= 255
    preds = conv_base.predict(x)
    print(preds)
    print(sum(preds))

generate_heatmaps(animals)


