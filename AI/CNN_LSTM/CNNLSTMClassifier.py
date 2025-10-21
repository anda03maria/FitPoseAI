import numpy as np
import tensorflow as tf
from numpy.f2py.crackfortran import verbose
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


from CNN.data_utils import evaluate
from CNN_LSTM.AbstractClassifierVideo import AbstractClassifierVideo

class CNNLSTMClassifier(AbstractClassifierVideo):

    def save(self, path):
        if hasattr(self, 'model'):
            self.model.save(path)
        else:
            raise AttributeError("Modelul nu a fost antrenat și nu poate fi salvat.")

    def build_cnnlstm_model(self, input_shape, time_steps):
        model = tf.keras.Sequential()

        # Aplica CNN pe fiecare cadru
        # folosim TimeDistributed pentru a aplica aceleasi layere CNN pe fiecare frame
        model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'),
                                         input_shape=(time_steps, *input_shape)))
        model.add(layers.TimeDistributed(layers.BatchNormalization()))
        model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

        model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
        model.add(layers.TimeDistributed(layers.BatchNormalization()))
        model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

        model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')))
        model.add(layers.TimeDistributed(layers.BatchNormalization()))
        model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

        model.add(layers.TimeDistributed(layers.Flatten()))

        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_classifier(self, train_gen, test_gen):
        # train_inputs = np.asarray(train_inputs)  # shape: (num_videos, time_steps, 128, 128, 3)
        # train_outputs = np.asarray(train_outputs)
        #
        # _, time_steps, h, w, c = train_inputs.shape

        X_sample, _ = train_gen[0]
        _, time_steps, h, w, c = X_sample.shape
        model = self.build_cnnlstm_model(input_shape=(h, w, c), time_steps=time_steps)

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # model.fit(train_inputs, train_outputs, epochs=50, batch_size=8, validation_split=0.1, verbose=2,
        #           callbacks=[early_stop])

        print("Train generator length (batches):", len(train_gen))
        print("Test generator length (batches):", len(test_gen))

        model.fit(train_gen,
                  validation_data=test_gen,
                  steps_per_epoch=len(train_gen),
                  validation_steps=len(test_gen),
                  epochs=25)

        #modif pentru SQUAT
        model.save('ohp.h5')
        return model

    def run_classifier(self, train_gen, test_gen):
        output_names = ["correct", "incorrect"]

        # train_inputs = np.asarray(train_inputs)
        # train_outputs = np.asarray(train_outputs)
        # test_inputs = np.asarray(test_inputs)
        # test_outputs = np.asarray(test_outputs)

        model = self.train_classifier(train_gen, test_gen)

        # computed_outputs = model.predict(test_inputs)
        # computed_outputs = np.round(computed_outputs)
        #
        # acc, precision, recall, conf_matrix = evaluate(test_outputs, computed_outputs, output_names)

        y_true = []
        y_pred = []

        for i in range(len(test_gen)):
            X_batch, y_batch = test_gen[i]
            if len(X_batch) == 0:
                continue
            preds = model.predict(X_batch)
            preds = np.round(preds).astype(int)
            y_true.extend(y_batch)
            y_pred.extend(preds)

        acc, precision, recall, conf_matrix = evaluate(np.array(y_true), np.array(y_pred), output_names)
        print("Accuracy: ", acc)
        print("Precision: ", precision)
        print("Recall: ", recall)

        cm = confusion_matrix(y_true, y_pred)
        print("\n=== Matrice de confuzie ===")
        print(cm)

        # F1-score
        f1 = f1_score(y_true, y_pred)
        print("F1-score:", round(f1, 2))

        # Optional: afișare grafică
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=output_names, yticklabels=output_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - CNN+LSTM")
        plt.tight_layout()
        plt.show()
