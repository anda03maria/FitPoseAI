from CNN_LSTM.CNNLSTMClassifier import CNNLSTMClassifier
from CNN_LSTM.AbstractClassifierVideo import AbstractClassifierVideo

def run_video_classification():
    # train_inputs, train_outputs, test_inputs, test_outputs = AbstractClassifierVideo.load_data()
    train_gen, test_gen = AbstractClassifierVideo.load_data()
    classifier = CNNLSTMClassifier()
    classifier.run_classifier(train_gen, test_gen)

if __name__ == '__main__':
    run_video_classification()