import matplotlib.pyplot as plt
import os, shutil
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import VGG16
import random

dirs = ['test','train','validation']
animals = ['zebra', 'buffalo', 'rhino', 'elephant']
base_dir = '/Users/robinvanderveken/Projects/Safari'


def split_files():
    animalfigs = {}
    for animal in animals:
        figlist = [name for name in os.listdir(os.path.join(base_dir,animal)) if name.split('.')[1]=='jpg']
        animalfigs.update({animal: figlist})

    for dir in dirs:
        os.mkdir(os.path.join(base_dir,dir))
        for animal in animals:
            os.mkdir(os.path.join(base_dir,dir,animal))
            no_figs = len(animalfigs[animal])
            if dir == 'train':
                figlist = animalfigs[animal][:2*int(no_figs/4)]
            elif dir == 'test':
                figlist = animalfigs[animal][2*int(no_figs/4):3*int(no_figs/4)]
            elif dir == 'validation':
                figlist = animalfigs[animal][3*int(no_figs / 4):]
            for fig in figlist:
                src = os.path.join(base_dir,animal,fig)
                dst = os.path.join(base_dir,dir,animal,fig)
                shutil.copyfile(src,dst)

def define_model(add_dropout=False):
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    if add_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(4,activation='softmax'))
    return model

model = define_model()
def load_pretrained_convnet():
    model = models.Sequential()
    conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

def visualizing_activations(animal='elephant',model=None):
    if model is None:
        model = define_model()
        model.load_weights('safari-wildlife/Model 1/safariv1.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])
    # get image
    imgs = os.listdir(os.path.join(test_dir,animal))
    img_path = os.path.join(test_dir,animal,imgs[random.randint(0,len(imgs))])
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    plt.imshow(img_tensor[0])
    # get outputs
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = tf.keras.models.Model(inputs=model.input,outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name,layer_activation in zip(layer_names,activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row

        display_grid = np.zeros((size*n_cols, images_per_row*size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:,:,col*images_per_row+row]
                channel_image-=channel_image.mean()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image,0,255).astype('uint8')
                display_grid[col*size: (col+1)*size,row*size:(row+1)*size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid,aspect='auto',cmap='viridis')

model = load_pretrained_convnet()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')



history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=7,
    validation_data=validation_generator,
    validation_steps=50
)

model.save(os.path.join(base_dir,'safari_VGG16.h5'))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()