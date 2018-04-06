import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
import pandas as pd
import scipy.ndimage
from skimage import measure, morphology
from skimage.transform import resize
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from glob import glob
import VisualizationHelper as vh

data_path = "C:\\Users\\burak\\Desktop\\New folder\\"
output_data = "images.npy"
g = glob(data_path + '/*.dcm')


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = (slope * image.astype(np.float64)).astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype = np.int16)


patient_scans = load_scan(data_path)
imgs = get_pixels_hu(patient_scans)

np.save(output_data, imgs)

imgs_to_process = np.load(output_data)

def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

imgs_after_resampling, spacing = resample(imgs_to_process, patient_scans, [1, 1, 1])

v, f = vh.make_mesh(imgs_after_resampling, 350)

