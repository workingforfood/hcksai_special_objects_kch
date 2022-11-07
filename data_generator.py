import os
import pickle
import random

import cv2
import keras
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import to_categorical


class Classifier_DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, train_data, img_shape=(128, 128), batch_size=64,
                 balance_classes=True, shuffle=False, istrain=True):
        np.random.seed(54321)
        self.data = train_data
        self.data_path = data_path
        self.img_size = img_shape
        self.batch_size = batch_size
        self.balance_classes = balance_classes
        self.shuffle = shuffle
        self.istrain = istrain
        self.current_epoch = 0
        self.classes = 8
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def on_epoch_end(self):
        if self.istrain:
            rnd = random.random() * 10000
            random.Random(rnd).shuffle(self.data)

    def generate_data(self, indexs):
        images_batch = []
        labels_batch = []

        for i in indexs:
            img_path = os.path.join(self.data_path, self.data[i]['filename'])
            labels_cl = to_categorical(self.data[i]['label'], self.classes)
            image = cv2.imread(img_path)
            if image is None:
                print(img_path)
            im_h, im_w, ch = image.shape

            # Augmentations
            if im_h > im_w:
                resize = iaa.Resize({"height": self.img_size[1], "width": 'keep-aspect-ratio'})
                crop = iaa.CropToFixedSize(width=self.img_size[0], height=random.randint(int(self.img_size[1] - self.img_size[1] * 0.02), self.img_size[1]))
            else:
                resize = iaa.Resize({"height": 'keep-aspect-ratio', "width": self.img_size[0]})
                crop = iaa.CropToFixedSize(width=random.randint(int(self.img_size[0] - self.img_size[0] * 0.02), self.img_size[0]), height=self.img_size[1])

            if not self.istrain:
                seq = iaa.Sequential([
                    resize,
                    iaa.PadToFixedSize(width=self.img_size[0], height=self.img_size[1], position='center', pad_cval=128)])
            else:
                seq = iaa.Sequential([
                    resize,
                    crop,
                    iaa.PadToFixedSize(width=self.img_size[0], height=self.img_size[1], position='center', pad_cval=(0, 255)),
                    iaa.Fliplr(0.5),
                    iaa.Sometimes(0.5, iaa.Crop(percent=0.05)),
                    iaa.Sometimes(0.7, iaa.Affine(rotate=(-45, 45), mode='edge', cval=(128, 255))),
                    iaa.OneOf([
                        iaa.Multiply((0.8, 1.1)),
                        iaa.LinearContrast(alpha=(0.9, 1.2))]),
                    iaa.Sometimes(0.6, iaa.AddToHueAndSaturation(value=[-15, 15], per_channel=True, from_colorspace="BGR")),
                    iaa.Sometimes(0.7, iaa.GaussianBlur(sigma=[0, 0.5])),
                    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02 * 255))),
                    iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.0, 0.5))),
                    iaa.Sometimes(0.5, iaa.JpegCompression(compression=(30, 55)))
                    ])

            seq_det = seq.to_deterministic()
            augmented_image = seq_det.augment_images([image])

            images_batch.append(augmented_image[0] / 255.)
            labels_batch.append(labels_cl)

        image_data = np.array(images_batch)
        labels_data = np.array(labels_batch)

        return image_data, labels_data

    def __getitem__(self, item):
        indexes = [i + item * self.batch_size for i in range(self.batch_size)]
        a, la = self.generate_data(indexes)

        return a, la


def create_val_set(labelsfile_path, val_set, val_split):
    with open(labelsfile_path, 'rb') as file:
        labels = pickle.load(file)
    np.random.seed(10101)
    np.random.shuffle(labels)
    np.random.seed(None)
    if val_set:
        num_val = int(len(labels) * val_split)
        num_train = len(labels) - num_val
        train_set = labels[:num_train]
        val_set = labels[num_train:]
        return train_set, val_set
    else:
        num_train = len(labels)
        train_set = labels[:num_train]
        val_set = None
        return train_set, val_set
