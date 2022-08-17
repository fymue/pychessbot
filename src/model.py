import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from pathlib import Path


class Model:
    def __init__(self, train_data_size=10000):
        self.train_data_size = train_data_size
        self.model = self.create_model()
        #self.data_path = "data/"
        self.data_path = str(Path(__file__).parents[1]) + "/data/" # path to data folder containing the training data
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(self.data_path)
    
    def load_data(self, data_path, test_size=0.2):
        data = np.load(data_path + "training_data_" + str(self.train_data_size) + ".npz")
        X, y = data["X"], data["y"]

        random_test_samples = np.random.choice(np.arange(X.shape[0]), int(test_size * X.shape[0]), replace=False)
        test_sample_mask = np.zeros(X.shape[0], dtype=bool)
        test_sample_mask[random_test_samples] = True

        X_test, y_test = X[test_sample_mask], y[test_sample_mask]
        X_train, y_train = X[~test_sample_mask], y[~test_sample_mask]

        """
        np.random.seed(1)
        np.random.shuffle(X_test)

        np.random.seed(1)
        np.random.shuffle(y_test)

        np.random.seed(2)
        np.random.shuffle(X_train)

        np.random.seed(2)
        np.random.shuffle(y_train)
        """

        return X_train, y_train, X_test, y_test

    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(layers.Flatten(input_shape=(6, 8, 8)))
        model.add(layers.Dense(256, activation="relu"))
        #model.add(layers.GaussianNoise(0.01))
        model.add(layers.Dense(128, activation="relu"))
        #model.add(layers.Dropout(0.1))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))


        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model
    
    def train(self, X_train=None, y_train=None):

        if not X_train: X_train = self.X_train
        if not y_train: y_train = self.y_train

        self.model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=2)
    
    def evaluate(self, X_test=None, y_test=None):

        if not X_test: X_test = self.X_test
        if not y_test: y_test = self.y_test

        res = self.model.evaluate(X_test, y_test, verbose=2)

        print(res)

        return res

if __name__ == "__main__":
    model = Model()
    model.train()
    model.evaluate()