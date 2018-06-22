import numpy as np
import json
import os
from random import shuffle

DATASET_INFO_PATH = "C:\\Users\\burak\\Desktop\\dataset\\25_9_p4_exclusive\\dataset.json"
DATASET_IMAGE_PATH = "C:\\Users\\burak\\Desktop\\dataset\\25_9_p4_exclusive\\images\\"
TRAINING_JSON = "C:\\Users\\burak\\Desktop\\dataset\\25_9_p4_exclusive\\training_dataset.json"
TEST_JSON = "C:\\Users\\burak\\Desktop\\dataset\\25_9_p4_exclusive\\test_dataset.json"

IMAGE_WIDTH = 25
IMAGE_HEIGHT = 25

def training_batch_generator(batch_size):
    training_data = json.load(open(TRAINING_JSON, "r"))
    print("size",len(training_data))
    for batch_index in range(int(len(training_data) / batch_size)):
        images = []
        labels = []
        locations = []
        for training_row in training_data[batch_index * batch_size:(batch_index + 1) * batch_size]:
            img = np.loadtxt(training_row['image_path'])
            images.append(img.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 1))

            location = []
            location.append(training_row['distance'])
            location.append(training_row['angle'])
            locations.append(location)

            if training_row['label'] == '1':
                labels.append([1, 0, 0])
            elif training_row['label'] == '2':
                labels.append([0, 1, 0])
            elif training_row['label'] == '3':
                labels.append([0, 0, 1])

        yield batch_index, np.array(images), np.array(labels), np.array(locations)


def test_batch_generator(batch_size):
    test_data = json.load(open(TEST_JSON, "r"))

    for batch_index in range(int(len(test_data) / batch_size)):
        images = []
        labels = []
        locations = []
        for test_row in test_data[batch_index * batch_size:(batch_index + 1) * batch_size]:
            img = np.loadtxt(test_row['image_path'])
            images.append(img.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 1))

            location = []
            location.append(test_row['distance'])
            location.append(test_row['angle'])
            locations.append(location)

            if test_row['label'] == '1':
                labels.append([1, 0, 0])
            elif test_row['label'] == '2':
                labels.append([0, 1, 0])
            elif test_row['label'] == '3':
                labels.append([0, 0, 1])

        yield batch_index, np.array(images), np.array(labels), np.array(locations)




def generate_train_test_dataset():

    with open(DATASET_INFO_PATH,"r") as f:
        datas = json.load(f)
        shuffle(datas)

        dataset = []
        for i in range(len(datas)):
            if datas[i]['label'] == '3' and i % 3 != 0:
                dataset.append(datas[i])
            elif datas[i]['label'] == '2' or datas[i]['label'] == '1':
                dataset.append(datas[i])

        training_dataset = dataset[: round(0.90 * len(dataset))]
        test_dataset = dataset[round(0.90 * len(dataset)):]
        training_file = open(TRAINING_JSON, "w")
        training_file.write(json.dumps(training_dataset, indent=4, sort_keys=True))
        training_file.close()
        test_file = open(TEST_JSON, "w")
        test_file.write(json.dumps(test_dataset, indent=4, sort_keys=True))
        test_file.close()


def get_train_data():
    training_data = json.load(open(TRAINING_JSON, "r"))
    # training_data = training_data[15034 * batch_size:]
    images = []
    labels = []
    locations = []
    for training_row in training_data:
        img = np.loadtxt(training_row['image_path'])
        images.append(img.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 1))

        location = []
        location.append(training_row['distance'])
        location.append(training_row['angle'])
        locations.append(location)

        if training_row['label'] == '1':
            labels.append([1, 0, 0])
        elif training_row['label'] == '2':
            labels.append([0, 1, 0])
        elif training_row['label'] == '3':
            labels.append([0, 0, 1])

    return np.array(images), np.array(labels), np.array(locations)


def get_valid_data():
    training_data = json.load(open(TRAINING_JSON, "r"))
    training_data = training_data[round(0.85 * len(training_data)):]
    images = []
    labels = []
    locations = []
    for training_row in training_data:
        img = np.loadtxt(training_row['image_path'])
        images.append(img.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 1))

        location = []
        location.append(training_row['distance'])
        location.append(training_row['angle'])
        locations.append(location)

        if training_row['label'] == '1':
            labels.append([1, 0, 0])
        elif training_row['label'] == '2':
            labels.append([0, 1, 0])
        elif training_row['label'] == '3':
            labels.append([0, 0, 1])

    return np.array(images), np.array(labels), np.array(locations)


def get_test_data():
    test_data = json.load(open(TEST_JSON, "r"))
    test_data = test_data[round(0.95 * len(test_data)):]
    images = []
    labels = []
    locations = []
    for test_row in test_data:
        img = np.loadtxt(test_row['image_path'])
        images.append(img.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 1))

        location = []
        location.append(test_row['distance'])
        location.append(test_row['angle'])
        locations.append(location)

        if test_row['label'] == '1':
            labels.append([1, 0, 0])
        elif test_row['label'] == '2':
            labels.append([0, 1, 0])
        elif test_row['label'] == '3':
            labels.append([0, 0, 1])

    return np.array(images), np.array(labels), np.array(locations)

def get_test_frame():
    test_data = json.load(open(TEST_JSON, "r"))
    test = test_data[1]
    print("label", test['label'])
    img = np.loadtxt(test['image_path'])
    img = img.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 1)
    return img

def main():
    generate_train_test_dataset()
    pass
    #a = np.loadtxt(DATASET_IMAGE_PATH +"1000001.txt")
    #VH.plot_slice(a)


if __name__ == '__main__':
    main()
