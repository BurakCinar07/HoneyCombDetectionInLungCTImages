import numpy as np
import json
import os
from random import shuffle
from PIL import Image

DATASET_INFO_PATH = "C:\\Users\\burak\\Desktop\\dataset\\dataset.json"
DATASET_IMAGE_PATH = "C:\\Users\\burak\\Desktop\\dataset\\images\\"
TRAINING_JSON = "C:\\Users\\burak\\Desktop\\dataset\\training_dataset.json"
TEST_JSON = "C:\\Users\\burak\\Desktop\\dataset\\test_dataset.json"

def training_batch_generator(batch_size):
    training_data = json.load(open(TRAINING_JSON, "r"))

    for batch_index in range(int(len(training_data) / batch_size)):
        images = []
        labels = []

        for training_row in training_data[batch_index * batch_size:(batch_index + 1) * batch_size]:
            img = np.loadtxt(training_row['image_path'])
            images.append(img)
            if training_row['label'] == '1':
                labels.append([1, 0, 0])
            elif training_row['label'] == '2':
                labels.append([0, 1, 0])
            elif training_row['label'] == '3':
                labels.append([0, 0, 1])

        yield batch_index, np.array(images), np.array(labels)


def test_batch_generator(batch_size):
    test_data = json.load(open(TEST_JSON, "r"))

    for batch_index in range(int(len(test_data) / batch_size)):
        images = []
        labels = []

        for test_row in test_data[batch_index * batch_size:(batch_index + 1) * batch_size]:
            img = np.loadtxt(test_row['image_path'])
            images.append(img)
            if test_row['label'] == '1':
                labels.append([1, 0, 0])
            elif test_row['label'] == '2':
                labels.append([0, 1, 0])
            elif test_row['label'] == '3':
                labels.append([0, 0, 1])

        yield batch_index, np.array(images), np.array(labels)

def generate_train_test_dataset():

    with open(DATASET_INFO_PATH,"r") as f:
        dataset = json.load(f)
        datas = []
        for data in dataset:
            img = np.loadtxt(data['image_path'])
            zero_count = np.count_nonzero(img==0)
            if zero_count < 185:
                datas.append(data)


        shuffle(datas)

        training_dataset = datas[: round(0.9 * len(datas))]
        test_dataset = datas[round(0.9 * len(datas)):]
        training_file = open("C:\\Users\\burak\\Desktop\\dataset\\training_dataset.json", "w")
        training_file.write(json.dumps(training_dataset, indent=4, sort_keys=True))
        training_file.close()
        test_file = open("C:\\Users\\burak\\Desktop\\dataset\\test_dataset.json", "w")
        test_file.write(json.dumps(test_dataset, indent=4, sort_keys=True))
        test_file.close()


def main():
    for batch_index, batch_imgs, batch_labels in test_batch_generator(20):
        print(batch_imgs)

if __name__ == '__main__':
    main()
