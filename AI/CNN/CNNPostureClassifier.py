from re import VERBOSE
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from CNN.AbstractClassifier import AbstractClassifier
from CNN.data_utils import evaluate
from CNN.utils import plot_histogram_data, plot_confusion_matrix

class CNNPostureClassifier(AbstractClassifier):

    def save(self, path):
        if hasattr(self, 'model'):
            self.model.save(path)
        else:
            raise AttributeError("Modelul nu a fost antrenat și nu poate fi salvat.")

    def train_classifier(self, train_inputs, train_outputs):
        classifier = tf.keras.Sequential()

        classifier.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        classifier.add(layers.BatchNormalization())
        classifier.add(layers.MaxPooling2D((2, 2)))

        classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
        classifier.add(layers.BatchNormalization())
        classifier.add(layers.MaxPooling2D((2, 2)))

        classifier.add(layers.Conv2D(128, (3, 3), activation='relu'))
        classifier.add(layers.BatchNormalization())
        classifier.add(layers.MaxPooling2D((2, 2)))

        classifier.add(layers.Flatten())
        classifier.add(layers.Dense(256, activation='relu'))
        classifier.add(layers.Dropout(0.5))
        classifier.add(layers.Dense(1, activation='sigmoid'))  # clasificare binary

        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        classifier.fit(train_inputs, train_outputs, epochs=30, batch_size=32, verbose=2)

        self.model = classifier
        classifier.save('barbellrow.h5')

        return classifier

    def run_classifier(self, train_inputs, train_outputs, test_inputs, test_outputs):
        output_names = ["correct", "incorrect"]

        train_inputs = np.asarray(train_inputs)
        train_outputs = np.asarray(train_outputs)
        test_inputs = np.asarray(test_inputs)
        test_outputs = np.asarray(test_outputs)

        plot_histogram_data(train_outputs, output_names, 'Posture Correct vs Incorrect')
        classifier = self.train_classifier(train_inputs, train_outputs)

        computed_outputs = classifier.predict(test_inputs)
        computed_outputs = np.round(computed_outputs)

        acc, precision, recall, conf_matrix = evaluate(test_outputs, computed_outputs, output_names)
        print("Accuracy: ", acc)
        print("Precision: ", precision)
        print("Recall: ", recall)

        plot_confusion_matrix(conf_matrix, output_names, "Posture CNN Classification")

