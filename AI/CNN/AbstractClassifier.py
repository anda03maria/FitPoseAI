import numpy as np
import tensorflow as tf
import os
import json
import cv2
import mediapipe as mp


class AbstractClassifier:

    @staticmethod
    def load_ids_from_splits(split_type):
        lumbar_path = os.path.join(os.path.dirname(__file__), '..', 'data_raw', 'BarbellRow', 'BarbellRow', 'Labeled_Dataset',
                               'Splits', 'Splits_Lumbar_Error',
                               f'{split_type}_ids.json')
        lumbar_path = os.path.abspath(lumbar_path)
        torso_path = os.path.join(os.path.dirname(__file__), '..', 'data_raw', 'BarbellRow', 'BarbellRow',
                                   'Labeled_Dataset',
                                   'Splits', 'Splits_TorsoAngle_Error',
                                             f'{split_type}_ids.json')
        torso_path = os.path.abspath(torso_path)

        ids = set()

        if os.path.exists(lumbar_path):
            with open(lumbar_path, 'r') as f:
                ids.update(json.load(f))

        if os.path.exists(torso_path):
            with open(torso_path, 'r') as f:
                ids.update(json.load(f))

        return ids

    @staticmethod
    def load_images_from_folder(folder, selected_ids):
        def preprocess_image(image_path):
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            image = image / 255.0
            return image

        images= []
        filenames = os.listdir(folder)
        for img_name in filenames:
            img_id = os.path.splitext(img_name)[0]
            if img_id in selected_ids:
                img_path = os.path.join(folder, img_name)
                img = preprocess_image(img_path)
                if img is not None:
                    images.append(img)
        return np.array(images)

    @staticmethod
    def load_data():
        train_ids = AbstractClassifier.load_ids_from_splits('train')
        test_ids = AbstractClassifier.load_ids_from_splits('test')

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        correct_folder = os.path.join(project_root, 'data_augmented', 'barbellrow', 'correct')
        incorrect_folder = os.path.join(project_root, 'data_augmented', 'barbellrow', 'incorrect')

        correct_train_images = AbstractClassifier.load_images_from_folder(correct_folder, train_ids)
        incorrect_train_images = AbstractClassifier.load_images_from_folder(incorrect_folder, train_ids)

        train_inputs = np.concatenate([correct_train_images, incorrect_train_images])
        train_outputs = [1] * len(correct_train_images) + [0] * len(incorrect_train_images)

        correct_test_images = AbstractClassifier.load_images_from_folder(correct_folder, test_ids)
        incorrect_test_images = AbstractClassifier.load_images_from_folder(incorrect_folder, test_ids)

        test_inputs = np.concatenate([correct_test_images, incorrect_test_images])
        test_outputs = [1] * len(correct_test_images) + [0] * len(incorrect_test_images)

        train_inputs = np.array(train_inputs)
        train_outputs = np.array(train_outputs)
        test_inputs = np.array(test_inputs)
        test_outputs = np.array(test_outputs)

        return train_inputs, train_outputs, test_inputs, test_outputs

    def train_classifier(self, train_inputs, train_outputs):
        raise NotImplementedError()

    def run_classifier(self, train_inputs, train_outputs, test_inputs, test_outputs):
        raise NotImplementedError()

