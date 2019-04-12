import glob
import os
import urllib.request
import zipfile

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#   There are 6 categories. Define constants
DMG_NONE = 0
DMG_INFRA = 1
DMG_NATUR = 2
FIRES = 3
FLOOD = 4
DMG_HUMAN = 5
categories = {'non_damage': 0,
              'damaged_infrastructure': 1,
              'damaged_nature': 2,
              'fires': 3,
              'flood': 4,
              'human_damage': 5}
classes = {'non_damage',
           'damaged_infrastructure',
           'damaged_nature',
           'fires',
           'flood',
           'human_damage'}


class DataSet:
    __slots__ = 'dataset_url', 'DOWNLOAD_CHUNK', 'data_file_name', 'dataset', 'train_datagen', 'train_generator'

    def __init__(self):
        self.dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00456/multimodal-deep-learning-disaster-response-mouzannar.zip'
        self.DOWNLOAD_CHUNK = 1024 * 1024
        self.data_file_name = 'multimodal-deep-learning-disaster-response-mouzannar.zip'
        self.dataset = []
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.train_generator = None

    def download(self):
        if not os.path.isfile(self.data_file_name):
            try:
                resp = urllib.request.urlopen(self.dataset_url)
                print('Downloading:', self.dataset_url)
                file_size = int(resp.headers['Content-Length'])
                with open(self.data_file_name, 'wb') as data_file:
                    for _ in tqdm(range(0, file_size, self.DOWNLOAD_CHUNK)):
                        data = resp.read(self.DOWNLOAD_CHUNK)
                        data_file.write(data)
            except:
                print('Failed to download.Cleaning up.')
                os.remove(self.data_file_name)
            print('Extracting.')
            zip_ref = zipfile.ZipFile(self.data_file_name, 'r')
            zip_ref.extractall('dataset')
            zip_ref.close()

    def load_data(self, target_size=(224, 224), batch_size=32):
        self.download()
        self.dataset = []
        for path in categories:
            this_catagory = glob.glob(os.path.join(os.path.join('dataset/multimodal', path), 'images/*.jpg'))
            print(path, ':', len(this_catagory))
            for file in this_catagory:
                tmp = np.zeros(6)
                tmp[categories[path]] = 1
                self.dataset.append((file, tmp))
        self.train_generator = self.train_datagen.flow_from_directory(
            'dataset/multimodal',
            target_size=target_size,
            color_mode='rgb',
            classes=classes,
            class_mode='categorical',
            batch_size=batch_size,
            # save_to_dir='tmp',
            interpolation='bicubic')
