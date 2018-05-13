import LungSegmentation as LS
import nibabel
import numpy as np
import VisualizationHelper as VH
import codecs, json
from scipy.ndimage import morphology

DATASET_OUTPUT_PATH = "C:\\Users\\burak\\PycharmProjects\\Bitirme\\dataset\\data_set.json"

p1_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT 2.nii"
p1_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
p2_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT.nii"
p2_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\11-5-2016 bt\\DICOM\\ST000000\\SE000001"
p3_nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\jakvalid.nii"
p3_dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Jak Valid Sevindiren\\DICOM\\S00001\\SER00002"

dataset = []

def generate_dataset(nifti_path, dicom_path):
    labeled_imgs = np.array(nibabel.load(nifti_path).get_data())
    labeled_imgs = np.transpose(labeled_imgs, (2, 1, 0))
    non_labeled_imgs = LS.load_scan(dicom_path)
    hu_value_imgs = LS.get_pixels_hu(non_labeled_imgs)[::-1]
    non_labeled_imgs = non_labeled_imgs[::-1]
    #ref_img = non_labeled_imgs[0]
    #resampled_imgs, spacing = LS.resample(hu_value_imgs, ref_img)
    #resampled_labeled_imgs = LS.resample_nifti(labeled_imgs, ref_img)
    #segmented_lungs = LS.segment_lung_mask(hu_value_imgs)

    binary_labeled_imgs = []
    for i in labeled_imgs:
        i[i == 2] = 1
        i[i == 3] = 1
        binary_labeled_imgs.append(i)

    binary_masks = LS.segment_lung_mask(hu_value_imgs)
    VH.plot_slice(binary_masks[29])
    binary_masks_with_veins = LS.segment_lung_mask(hu_value_imgs, fill_lung_structures=False)
    VH.plot_slice(binary_masks_with_veins[29])
    VH.plot_slice(binary_masks[29] - binary_masks_with_veins[29])
    VH.plot_slice(hu_value_imgs[29] * binary_masks_with_veins[29])
    dices = []
    for i in range(24,48):
        dice_metrix = LS.dice_metric_coeffecient(binary_labeled_imgs[i], binary_masks[i])
        dices.append(dice_metrix)
        print(dice_metrix)

    print("Dice metrix average for patient:", sum(dices)/len(dices))
    #binary_masks = LS.segment_lung_mask(resampled_imgs)
    #binary_masks = morphology.binary_fill_holes(morphology.binary_dilation(morphology.binary_fill_holes(binary_masks > 0), iterations=4))
    #segmented_lungs = binary_masks * resampled_imgs
    """
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
    """
generate_dataset(p1_nifti_path, p1_dicom_path)
#generate_dataset(p2_nifti_path, p2_dicom_path)
#generate_dataset(p3_nifti_path, p3_dicom_path)
#json.dump(dataset, codecs.open(DATASET_OUTPUT_PATH, 'w+', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
