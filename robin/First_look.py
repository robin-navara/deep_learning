# import
from keras.datasets import mnist

# data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
len(train_labels)
train_labels

test_images.shape