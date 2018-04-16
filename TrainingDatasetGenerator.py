import nibabel
import LungSegmentation as LS
import nibabel
from glob import glob
import numpy as np
import VisualizationHelper as VH
from PIL import Image

nifti_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\isaretlemeler\\HRCT 2.nii"
dicom_path = "C:\\Users\\burak\\Desktop\\Bitirme\\Kaynaklar\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"

labeled_imgs = np.array(nibabel.load(nifti_path).get_data())
non_labeled_imgs = LS.load_scan(dicom_path)
hu_value_imgs = LS.get_pixels_hu(non_labeled_imgs)[::-1]
non_labeled_imgs = non_labeled_imgs[::-1]

#segmented_lungs = []
#for img in hu_value_imgs:
#    segmented_lungs.append(LS.make_lungmask(img))

segmented_lungs = LS.segment_lung_mask(hu_value_imgs)

#imaqw = segmented_lungs[25]
#limag = labeled_imgs[:,:,25]
counts = np.zeros(2)

def crop_lung(img, row, col):
    cropped = np.zeros((3,3), dtype=np.float)
    if(img[row][col] == 0.):
        counts[0] +=1

    counts[1] += 1
    for i in range(2):
        for j in range(2):
            cropped[i][j] = img[row - 1 + i][col - 1 + j]

    return cropped


for k in range(len(segmented_lungs)):
    image = segmented_lungs[k]
    VH.plot_slice(image)
    for i in range(labeled_imgs.shape[0]):
        for j in range(labeled_imgs.shape[1]):
            if labeled_imgs[i][j][k] == 1:
                print("slice, i, j, label", k, i, j, labeled_imgs[i][j][k])
                cropped_img = crop_lung(image, j,i)
                #print(cropped_img)
            elif labeled_imgs[i][j][k] == 2:
                print("slice, i, j, label", k, i, j, labeled_imgs[i][j][k])
                cropped_img = crop_lung(image, j,i)
                #print(cropped_img)
            elif labeled_imgs[i][j][k] == 3:
                print("slice, i, j, label", k, i, j, labeled_imgs[i][j][k])
                cropped_img = crop_lung(image, j,i)
                #print(cropped_img)


print("Loss" , counts[0], counts[1], counts[0]/counts[1])