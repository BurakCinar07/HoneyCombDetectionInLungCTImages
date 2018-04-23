import LungSegmentation as LS
import nibabel
from glob import glob
import numpy as np
import VisualizationHelper as VH
import codecs, json
import scipy
from scipy.ndimage import morphology

DATASET_OUTPUT_PATH = "C:\\Users\\burak\\PycharmProjects\\Bitirme\\dataset\\data_set.json"

p1_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT 2.nii"
p1_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
p2_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT.nii"
p2_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\11-5-2016 bt\\DICOM\\ST000000\\SE000001"
p3_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\jakvalid.nii"
p3_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Jak Valid Sevindiren\\DICOM\\S00001\\SER00002"

dataset = []

def crop_lung(img, row, col, label):
    cropped = np.zeros((3,3), dtype=np.float)
    if(img[row][col] == 0.):
        return

    for i in range(3):
        for j in range(3):
            cropped[i][j] = img[row - 1 + i][col - 1 + j]

    dataset.append({
        'image': cropped.tolist(),
        'label': label
    })


def generate_dataset(nifti_path, dicom_path):
    labeled_imgs = np.array(nibabel.load(nifti_path).get_data())
    labeled_imgs = np.transpose(labeled_imgs, (2, 1, 0))
    non_labeled_imgs = LS.load_scan(dicom_path)
    hu_value_imgs = LS.get_pixels_hu(non_labeled_imgs)[::-1]
    non_labeled_imgs = non_labeled_imgs[::-1]

    ref_img = non_labeled_imgs[0]
    resampled_imgs, spacing = LS.resample(hu_value_imgs, ref_img)
    #resampled_labeled_imgs = LS.resample_nifti(labeled_imgs, ref_img)

    test_mask = LS.segment_lung_mask(resampled_imgs)
    binary_masks = morphology.binary_fill_holes(morphology.binary_dilation(morphology.binary_fill_holes(test_mask > 0), iterations=4))
    segmented_lungs = binary_masks * resampled_imgs
    for i in range(15, len(segmented_lungs)):
        VH.plot_slice(segmented_lungs[i])
    """
    binary_masks = LS.segment_lung_mask(resampled_imgs)
    binary_masks = morphology.binary_fill_holes(morphology.binary_dilation(morphology.binary_fill_holes(binary_masks > 0), iterations=4))
    segmented_lungs = binary_masks * resampled_imgs
    VH.plot_slice(segmented_lungs[20])
    
    for k in range(len(segmented_lungs)):
        image = segmented_lungs[k]
        if 1 in labeled_imgs[:, :, k] or 2 in labeled_imgs[:, :, k] or 3 in labeled_imgs[:, :, k]:
            for i in range(labeled_imgs.shape[0]):
                for j in range(labeled_imgs.shape[1]):
                    if labeled_imgs[i][j][k] == 1:
                        print("slice, i, j, label", k, i, j, labeled_imgs[i][j][k])
                        crop_lung(image, i, j, label=1)
                    elif labeled_imgs[i][j][k] == 2:
                        print("slice, i, j, label", k, i, j, labeled_imgs[i][j][k])
                        crop_lung(image, i, j,label=2)
                    elif labeled_imgs[i][j][k] == 3:
                        print("slice, i, j, label", k, i, j, labeled_imgs[i][j][k])
                        crop_lung(image, i, j,label=3)

"""
generate_dataset(p1_nifti_path, p1_dicom_path)
#generate_dataset(p2_nifti_path, p2_dicom_path)
#generate_dataset(p3_nifti_path, p3_dicom_path)
json.dump(dataset, codecs.open(DATASET_OUTPUT_PATH, 'w+', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
