from CNN.AbstractClassifier import AbstractClassifier
from CNN.CNNPostureClassifier import CNNPostureClassifier
import numpy as np

import os

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

def run_barbellrow_classification():
    # 1. Load train/test data conform Splits
    train_inputs, train_outputs, test_inputs, test_outputs = AbstractClassifier.load_data()

    # 2. Optional: vezi distribuția claselor (corect/incorect)
    # print(f"Train set: {len(train_inputs)} imagini")
    # print(f"Test set: {len(test_inputs)} imagini")

    # 3. Train și evaluate CNN
    cnn_classifier = CNNPostureClassifier()
    cnn_classifier.run_classifier(train_inputs, train_outputs, test_inputs, test_outputs)

if __name__ == '__main__':
    run_barbellrow_classification()
