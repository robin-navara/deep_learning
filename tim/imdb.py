from keras.datasets import imdb
from keras import models, layers, optimizers
import numpy as np
import random


def decode_review(review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value,key) for (key,value) in word_index.items()]
    )
    decoded_review = ' '.join(
        [reverse_word_index.get(i-3,'?') for i in review]
    )
    return decoded_review


def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results


def pull_ambiguous_review():
    preds = list(model.predict(x_test).flatten())
    ambilist = [i for i in range(len(predlist)) if predlist[i] > .4 and predlist[i] < .6]
    ambichoice = random.choice(ambilist)
    review = decode_review(train_data[ambichoice])
    pred = preds[ambichoice]
    if train_labels[ambichoice]:
        rating = 'Positive'
    else:
        rating = 'Negative'
    print('The model finds the following '+rating+' review ambigous with a score of '+str(pred)+'.')
    print(review)
    return None

if __name__ = "__main__":
    #load data
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    #define model
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    #train model
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    #validate model by user
    for i in range(3):
        pull_ambiguous_review()
    