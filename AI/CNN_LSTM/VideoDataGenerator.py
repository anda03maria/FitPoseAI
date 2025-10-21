import numpy as np
from tensorflow.keras.utils import Sequence



class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size=8, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.video_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        from CNN_LSTM.AbstractClassifierVideo import AbstractClassifierVideo
        index = int(index)  # asigurare
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # batch_video_paths = [self.video_paths[k] for k in batch_indexes]
        # batch_labels = [self.labels[k] for k in batch_indexes]
        #
        # X = [AbstractClassifierVideo.load_video_as_frames(path) for path in batch_video_paths]
        batch_video_paths = [self.video_paths[int(k)] for k in batch_indexes]
        batch_labels = [self.labels[int(k)] for k in batch_indexes]
        X = [AbstractClassifierVideo.load_video_as_frames(path) for path in batch_video_paths]
        return np.array(X), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)