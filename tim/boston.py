from keras.datasets import boston_housing
from keras import models, layers

import numpy as np

(train_data,train_targets), (test_data,test_targets) = boston_housing.load_data()

#normalize data
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data-mean)/std
test_data = (test_data-mean)/std

def build_model(dimension_mid_layer=64):
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(dimension_mid_layer,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

def k_fold_validation(k=4,num_epochs=100,dim_middle_layer=64):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []
    dim_middle_layer = 64
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples, :]
        val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples, :],
             train_data[(i + 1) * num_val_samples:, :]],
            axis=0
        )
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0
        )

        model = build_model(dim_middle_layer)
        model.fit(partial_train_data, partial_train_targets,
                  epochs=num_epochs, batch_size=1, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
    return model, all_scores

if __name__ == '__main__':
    model, all_scores = k_fold_validation(k=4,num_epochs=500)
    print(all_scores)

