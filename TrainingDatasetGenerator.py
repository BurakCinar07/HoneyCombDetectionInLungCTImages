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

test_dicom = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Ünal\\DICOM\\ST000000\\SE000005\\"



dataset = []
loss = np.zeros(2)
def crop_lung(img, row, col, label):
    cropped = np.zeros((3,3), dtype=np.float)
    if(img[row][col] == 0.):
        loss[0] += 1

    loss[1] += 1
    print(loss[0], loss[1])
    #dataset.append({
    #    'image': cropped.tolist(),
    #    'label': label
    #})


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

    for k in range(len(segmented_lungs)):
        image = segmented_lungs[k]
        labeled_img = resampled_labeled_imgs[k]
        if 1 in labeled_img or 2 in labeled_img or 3 in labeled_img:
            for i in range(labeled_img.shape[0]):
                for j in range(labeled_img.shape[1]):
                    if labeled_img[i][j] == 1:
                        crop_lung(image, i, j, label=1)
                    elif labeled_img[i][j] == 2:
                        crop_lung(image, i, j,label=2)
                    elif labeled_img[i][j] == 3:
                        crop_lung(image, i, j,label=3)
    
generate_dataset(p1_nifti_path, p1_dicom_path)

generate_dataset(p2_nifti_path, p2_dicom_path)
generate_dataset(p3_nifti_path, p3_dicom_path)
#json.dump(dataset, codecs.open(DATASET_OUTPUT_PATH, 'w+', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
print("Final loss", loss[0], loss[1], loss[0]/loss[1])
