import os
import json
import cv2
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from CNN_LSTM.VideoDataGenerator import VideoDataGenerator


class AbstractClassifierVideo:
    FRAME_COUNT = 20
    FRAME_SIZE = (64, 64)

    @staticmethod
    def load_ids_from_splits(split_type):
        #modif pentru SQUAT
        # generic_path = f'/content/drive/MyDrive/DateLicenta/splits/Splits_OHP/{split_type}_keys.json'
        # cv_path = os.path.join(os.path.dirname(__file__), '..', 'data_raw', 'Squat', 'Squat', 'Labeled_Dataset', 'Splits',
        #                        f'{split_type}_keys.json')
        cv_path = os.path.join(os.path.dirname(__file__), '..', 'data_raw', 'OHP', 'OHP', 'Labeled_Dataset',
                               'Splits',
                               f'{split_type}_keys.json')
        cv_path = os.path.abspath(cv_path)

        ids = set()
        #if os.path.exists(cv_path):
        with open(cv_path, 'r') as f:
            ids.update(json.load(f))
        return ids

    @staticmethod
    def load_video_as_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // AbstractClassifierVideo.FRAME_COUNT)

        count = 0
        while len(frames) < AbstractClassifierVideo.FRAME_COUNT and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame = cv2.resize(frame, AbstractClassifierVideo.FRAME_SIZE)
                frame = frame / 255.0  # normalize
                frames.append(frame)
            count += 1

        cap.release()
        # completeaza cu cadre negre
        while len(frames) < AbstractClassifierVideo.FRAME_COUNT:
            frames.append(np.zeros((64, 64, 3)))
        return np.array(frames)

    @staticmethod
    def load_videos_from_folder(folder, selected_ids):
        videos = []
        filenames = os.listdir(folder)
        for video_name in filenames:
            video_id = os.path.splitext(video_name)[0]
            if video_id in selected_ids:
                video_path = os.path.join(folder, video_name)
                frames = AbstractClassifierVideo.load_video_as_frames(video_path)
                videos.append(frames)

        return videos

    @staticmethod
    def load_data(batch_size=8):
        train_ids = AbstractClassifierVideo.load_ids_from_splits('train')
        test_ids = AbstractClassifierVideo.load_ids_from_splits('test')

        #modif pentru SQUAT
        # correct_folder = "/content/drive/MyDrive/DateLicenta/ohp/correct"
        # incorrect_folder = "/content/drive/MyDrive/DateLicenta/ohp/incorrect"
        # correct_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'squat', 'correct')
        # incorrect_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'squat', 'incorrect')

        correct_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'ohp', 'correct')
        incorrect_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'ohp', 'incorrect')
        correct_folder = os.path.abspath(correct_folder)
        incorrect_folder = os.path.abspath(incorrect_folder)

        def get_video_paths_and_labels(folder, ids, label):
            paths, labels = [], []
            for filename in os.listdir(folder):
                video_id = os.path.splitext(filename)[0]
                if video_id in ids:
                    full_path = os.path.join(folder, filename)
                    paths.append(full_path)
                    labels.append(label)
            return paths, labels

        correct_train_paths, correct_train_labels = get_video_paths_and_labels(correct_folder, train_ids, 1)
        incorrect_train_paths, incorrect_train_labels = get_video_paths_and_labels(incorrect_folder, train_ids, 0)
        train_video_paths = correct_train_paths + incorrect_train_paths
        train_labels = correct_train_labels + incorrect_train_labels

        correct_test_paths, correct_test_labels = get_video_paths_and_labels(correct_folder, test_ids, 1)
        incorrect_test_paths, incorrect_test_labels = get_video_paths_and_labels(incorrect_folder, test_ids, 0)
        test_video_paths = correct_test_paths + incorrect_test_paths
        test_labels = correct_test_labels + incorrect_test_labels

        train_gen = VideoDataGenerator(train_video_paths, train_labels, batch_size=batch_size)
        test_gen = VideoDataGenerator(test_video_paths, test_labels, batch_size=batch_size)

        return train_gen, test_gen

        # correct_train = AbstractClassifierVideo.load_videos_from_folder(correct_folder, train_ids)
        # incorrect_train = AbstractClassifierVideo.load_videos_from_folder(incorrect_folder, train_ids)
        # train_inputs = correct_train + incorrect_train
        # train_outputs = [1] * len(correct_train) + [0] * len(incorrect_train)
        #
        # correct_test = AbstractClassifierVideo.load_videos_from_folder(correct_folder, test_ids)
        # incorrect_test = AbstractClassifierVideo.load_videos_from_folder(incorrect_folder, test_ids)
        # test_inputs = correct_test + incorrect_test
        # test_outputs = [1] * len(correct_test) + [0] * len(incorrect_test)

        # return np.array(train_inputs), np.array(train_outputs), np.array(test_inputs), np.array(test_outputs)

    def train_classifier(self, train_inputs, train_outputs):
        raise NotImplementedError()

    def run_classifier(self, train_inputs, train_outputs, test_inputs, test_outputs):
        raise NotImplementedError()