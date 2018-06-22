import LungSegmentation as LS
import nibabel
import numpy as np
import VisualizationHelper as VH
import codecs, json
import math
from scipy.ndimage import morphology


# TODO: window size ı tek sayı yap
# TODO: coreları yan yana kaydır. croplanmış resimlerin akciğer merkezine uzaklıklarını hesapla
WINDOW_SIZE = 25
CORE_OFFSET = 5

DATASET_IMAGE_PATH = "C:\\Users\\burak\\Desktop\\dataset\\25_9_p4_exclusive\\images\\"
DATASET_JSON_PATH = "C:\\Users\\burak\\Desktop\\dataset\\25_9_p4_exclusive\\dataset.json"
p1_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT 2.nii"
p1_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
p2_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT.nii"
p2_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\11-5-2016 bt\\DICOM\\ST000000\\SE000001"
p3_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\jakvalid.nii"
p3_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Jak Valid Sevindiren\\DICOM\\S00001\\SER00002"
p4_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\3a"
p4_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\3a.nii"
index = 0
dataset = []



def find_most_frequent_label(labeled_image, mid):
    img = []
    for i in range(labeled_image.shape[0]):
        for j in range(labeled_image.shape[1]):
            if j < mid + CORE_OFFSET  and j > mid - CORE_OFFSET and \
                    i < mid + CORE_OFFSET and i > mid - CORE_OFFSET:
                img.append(labeled_image[i][j])

    zero_count = np.count_nonzero((np.array(img) == 0))
    if zero_count / len(img) > 0.80:
        return 0
    (values, counts) = np.unique(np.array(img), return_counts=True)
    ind = np.argmax(counts)
    if values[ind] == 0:
        ind = 1 + np.argmax(counts[1:])

    return values[ind]



def crop_image(image, labeled_image, row, col):
    middle_point = int(WINDOW_SIZE / 2)
    cropped_labeled_img = labeled_image[row - middle_point: row + middle_point + 1,
                          col - middle_point: col + middle_point + 1]

    cropped_image = image[row - middle_point: row + middle_point + 1,
                          col - middle_point: col + middle_point + 1]

    label = find_most_frequent_label(cropped_labeled_img, middle_point)
    global index
    index = index + 1
    path = DATASET_IMAGE_PATH + str(index) + ".txt"
    if label != 0:
        np.savetxt(path, cropped_image)
    return path, label


def generate_dataset(nifti_path, dicom_path):
    labeled_imgs = np.array(nibabel.load(nifti_path).get_data())
    labeled_imgs = np.transpose(labeled_imgs, (2, 1, 0))
    non_labeled_imgs = LS.load_scan(dicom_path)
    hu_value_imgs = LS.get_pixels_hu(non_labeled_imgs)[::-1]

    masks = LS.segment_lung_mask(hu_value_imgs)
    binary_masks = morphology.binary_fill_holes(
        morphology.binary_dilation(
            morphology.binary_fill_holes(masks > 0),
            iterations=1)
    )
    segmented_lungs = hu_value_imgs * binary_masks


    # TODO: sliding windowu overlap etmicek şekilde coreların aralarında boşluk kalmıcak şekilde
    for k in range(len(segmented_lungs)):
        image = segmented_lungs[k]
        labeled_img = labeled_imgs[k]
        if 1 in labeled_img or 2 in labeled_img or 3 in labeled_img:
            for i in range(0, labeled_img.shape[1], 3):
                for j in range(0, labeled_img.shape[0], 4):
                    if labeled_img[i][j] != 0:
                        path, label = crop_image(image, labeled_img, i, j)
                        if label != 0:
                            dataset.append({
                                'image_path': path,
                                'label': str(label)
                            })


generate_dataset(p1_nifti_path, p1_dicom_path)
generate_dataset(p2_nifti_path, p2_dicom_path)
generate_dataset(p3_nifti_path, p3_dicom_path)
#generate_dataset(p4_nifti_path, p4_dicom_path)
with open(DATASET_JSON_PATH, 'w') as file:
    file.write(json.dumps(dataset, indent=4, sort_keys=True))

import datasource as ds

#ds.generate_train_test_dataset()
# json.dump(dataset, codecs.open(DATASET_OUTPUT_PATH, 'w+', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
