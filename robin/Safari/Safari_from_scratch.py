import matplotlib.pyplot as plt
import os
from keras import layers, models, optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.applications.vgg16 import VGG16

def main(output_name = 'safari_from_scratch',model_type='base',epochs=100,lr=1e-4,decay=0,momentum=0):

    # base directories for images
    base_dir = '/Users/robinvanderveken/Projects/Safari/Subsets'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # initialize model
    if model_type == 'base':
        model = safari_model_base(learning_rate=lr,decay=decay,momentum=momentum)
    elif model_type == 'vgg16':
        model = safari_model_vgg16(learning_rate=lr, decay=decay, momentum=momentum)

    ## TRAIN ##
    #image generator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    #actual images (train
    train_generator = train_datagen.flow_from_directory(
        train_dir,target_size=(150, 150),batch_size=32,class_mode='categorical'
    )

    ## TEST AND VALIDATION ##
    #image generator
    test_validation_datagen = ImageDataGenerator(rescale=1. / 255)

    #actual images (train)
    test_generator = test_validation_datagen.flow_from_directory(
        test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
    )

    # actual images (validation)
    validation_generator = test_validation_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
    )

    # base directory for output
    output_dir = '/Users/robinvanderveken/PycharmProjects/deep_learning/robin/Safari'

    # path to save checkpoints
    checkpoint_path = os.path.join(output_dir,'trainCheckpoints',output_name+'.ckpt')

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=True)

    ## TRAINING THE MODEL ##
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[cp_callback]
    )

    # save dictionary object history (can be loaded back with pickle.load())
    history_output = os.path.join(output_dir,'trainHistoryDict',output_name)
    with open(history_output,
              'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # save model output
    model_output_file = os.path.join(output_dir,'ModelOutput',output_name+'.h5')
    model.save(model_output_file)


def safari_model_base(learning_rate=1e-4,decay=0,momentum=0):
    model = models.Sequential()
    model.add(layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(4,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=learning_rate,decay=decay,momentum=momentum),
                  metrics=['acc'])

    return model

def safari_model_vgg16(learning_rate=1e-4,decay=0,momentum=0):
    model = models.Sequential()
    conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=learning_rate,decay=decay,momentum=momentum),
                  metrics=['acc'])

    return model

def read_history(file_name,plot=True):
    base_dir = '/Users/robinvanderveken/PycharmProjects/deep_learning/robin/Safari/trainHistoryDict'
    with open(os.path.join(base_dir,file_name),'rb') as file_pi:
        history = pickle.load(file_pi)

    if plot == True:
        acc = history['acc']
        val_acc = history['val_acc']
        loss = history['loss']
        val_loss = history['val_loss']
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

    return history

def load_model(file_name,summary=True):
    base_dir = '/Users/robinvanderveken/PycharmProjects/deep_learning/robin/Safari/ModelOutput'
    model = models.load_model(os.path.join(base_dir,file_name+'.h5'))

    if summary==True:
        print(model.summary())

    return model

if __name__ == "__main__":
    main()
