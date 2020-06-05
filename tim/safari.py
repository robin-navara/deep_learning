import os, shutil
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


dirs = ['test','train','validation']
animals = ['zebra', 'buffalo', 'rhino', 'elephant']
base_dir = '/Users/tim/Documents/Tutorials/Deep Learning in Python/deep_learning/tim/safari-wildlife'


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

model = define_model(True)
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

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save(os.path.join(base_dir,'safariv2.h5'))

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