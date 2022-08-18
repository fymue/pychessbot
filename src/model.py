import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from pathlib import Path


class Model:
    def __init__(self, train_data_size=10000):
        # set data paths, create the model and load the data

        self.train_data_size = train_data_size

        self.parent_path = str(Path(__file__).parents[1]) 
        self.data_path = self.parent_path + "/data/" # path to data folder containing the training data
        self.model_path = self.parent_path + "/model/"

        self.model = self.create_model()
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(self.data_path)
    
    def load_data(self, data_path, test_size=0.2):
        # load the data generated by the PGNParser class
        
        data = np.load(data_path + "training_data_" + str(self.train_data_size) + ".npz")
        X, y = data["X"], data["y"]

        # randomly draw test_size of the total samples as a test set
        random_test_samples = np.random.choice(np.arange(X.shape[0]), int(test_size * X.shape[0]), replace=False)
        test_sample_mask = np.zeros(X.shape[0], dtype=bool)
        test_sample_mask[random_test_samples] = True

        X_test, y_test = X[test_sample_mask], y[test_sample_mask]
        X_train, y_train = X[~test_sample_mask], y[~test_sample_mask]

        return X_train, y_train, X_test, y_test

    def create_model(self):
        # create a neural network with 1 output neuron
        # (determines if the current board state resulted
        # from a good move or not)

        model = tf.keras.models.Sequential()

        model.add(layers.Conv2D(16, (2, 2), activation="relu", padding="same", input_shape=(6, 8, 8)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (2, 2), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        #model.add(layers.Dense(128, activation="relu"))
        #model.add(layers.Dropout(0.1))
        model.add(layers.Dense(64, activation="relu"))
        #model.add(layers.Dropout(0.1))
        model.add(layers.Dense(8, activation="relu"))
        #model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation="sigmoid"))


        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model
    
    def train(self, X_train=None, y_train=None):
        # train the neural nework

        if not X_train: X_train = self.X_train
        if not y_train: y_train = self.y_train

        self.model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, shuffle=True, verbose=2)
    
    def evaluate(self, X_test=None, y_test=None):
        # test the accuracy of the trained neural network

        if not X_test: X_test = self.X_test
        if not y_test: y_test = self.y_test

        res = self.model.evaluate(X_test, y_test, verbose=2)

        print(res)
    
    def save(self, name):
        self.model.save(self.model_path + name)

if __name__ == "__main__":
    model = Model(train_data_size=1000000)
    model.train()
    model.evaluate() # so far: training, validation and test accuracy all around 61%
    model.save("chess_model")