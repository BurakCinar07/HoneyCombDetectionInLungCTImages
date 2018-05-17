import LungSegmentation as LS
import nibabel
import numpy as np
import VisualizationHelper as VH
import codecs, json
from collections import Counter
import pickle
from PIL import Image

WINDOW_SIZE = 7

DATASET_OUTPUT_PATH = "C:\\Users\\burak\\PycharmProjects\\Bitirme\\dataset\\data_set2.pkl"
DATASET_IMAGE_PATH = "C:\\Users\\burak\\Desktop\\dataset\\images\\"
DATASET_JSON_PATH = "C:\\Users\\burak\\Desktop\\dataset\\dataset.json"
p1_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT 2.nii"
p1_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
p2_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT.nii"
p2_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\11-5-2016 bt\\DICOM\\ST000000\\SE000001"
p3_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\jakvalid.nii"
p3_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Jak Valid Sevindiren\\DICOM\\S00001\\SER00002"

index = 0

def find_most_frequent_label(labeled_image):
    (values, counts) = np.unique(labeled_image, return_counts=True)
    if counts[0] / sum(counts) > 0.85:
        return 0
    ind = np.argmax(counts)
    if values[ind] == 0:
        ind = 1 + np.argmax(counts[1:])
    return values[ind]


def crop_image(image, labeled_image, row, col):
    cropped_labeled_img = labeled_image[row - WINDOW_SIZE: row + WINDOW_SIZE, col - WINDOW_SIZE: col + WINDOW_SIZE]
    cropped_image = image[row - WINDOW_SIZE: row + WINDOW_SIZE, col - WINDOW_SIZE: col + WINDOW_SIZE]
    label = find_most_frequent_label(cropped_labeled_img)
    global index
    index = index + 1
    path = DATASET_IMAGE_PATH + str(index) + ".txt"
    np.savetxt(path, cropped_image)
    return path, label


def generate_dataset(nifti_path, dicom_path):
    labeled_imgs = np.array(nibabel.load(nifti_path).get_data())
    labeled_imgs = np.transpose(labeled_imgs, (2, 1, 0))
    non_labeled_imgs = LS.load_scan(dicom_path)
    hu_value_imgs = LS.get_pixels_hu(non_labeled_imgs)[::-1]
    non_labeled_imgs = non_labeled_imgs[::-1]

    ref_img = non_labeled_imgs[0]
    resampled_imgs, spacing = LS.resample(hu_value_imgs, ref_img)
    resampled_labeled_imgs = LS.resample_nifti(labeled_imgs, ref_img)

    binary_masks = LS.segment_lung_mask(resampled_imgs)
    segmented_lungs = binary_masks * resampled_imgs
    with open(DATASET_JSON_PATH, 'a') as file:
        for k in range(len(segmented_lungs)):
            image = segmented_lungs[k]
            labeled_img = resampled_labeled_imgs[k]
            if 1 in labeled_img or 2 in labeled_img or 3 in labeled_img:
                for i in range(0, labeled_img.shape[1], 5):
                    dataset = []
                    for j in range(labeled_img.shape[0]):
                        if labeled_img[i][j] != 0:
                            path, label = crop_image(image, labeled_img, i, j)
                            if label != 0:
                                dataset.append({
                                    'image_path': path,
                                    'label': str(label)
                                })
                    file.write(json.dumps(dataset, indent=4, sort_keys=True))


generate_dataset(p1_nifti_path, p1_dicom_path)
generate_dataset(p2_nifti_path, p2_dicom_path)
generate_dataset(p3_nifti_path, p3_dicom_path)
#json.dump(dataset, codecs.open(DATASET_OUTPUT_PATH, 'w+', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
