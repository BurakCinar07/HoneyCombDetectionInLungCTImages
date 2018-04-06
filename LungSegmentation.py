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

data_path = "C:\\Users\\burak\\Desktop\\New folder\\"
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


print(len(load_scan(data_path)))