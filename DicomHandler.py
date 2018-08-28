import os
import png
import pydicom
import numpy as np


def mri_to_png(mri_file):
    """ Function to convert from a DICOM image to png
        @param mri_file: An opened file like object to read te dicom data
        @param png_file: An opened file like object to write the png data
    """

    # Extracting data from the mri file
    plan = pydicom.read_file(mri_file)
    shape = plan.pixel_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = plan.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Writing the PNG file
    #im = png.Writer(shape[1], shape[0], greyscale=True)
    return image_2d_scaled

def convert_file(mri_file_path):
    """ Function to convert an MRI binary file to a
        PNG image file.
        @param mri_file_path: Full path to the mri file
        @param png_file_path: Fill path to the png file
    """

    if not os.path.exists(mri_file_path):
        raise Exception('File "%s" does not exists' % mri_file_path)


    mri_file = open(mri_file_path, 'rb')

    img = mri_to_png(mri_file)
    return img


def convert_folder(mri_folder):
    """ Convert all MRI files in a folder to png files
        in a destination folder
    """

    # Create the folder for the pnd directory structure
    #os.makedirs(png_folder)
    images = []
    # Recursively traverse all sub-folders in the path
    for mri_sub_folder, subdirs, files in os.walk(mri_folder):
        for mri_file in os.listdir(mri_sub_folder):
            mri_file_path = os.path.join(mri_sub_folder, mri_file)

            # Make sure path is an actual file
            if os.path.isfile(mri_file_path):

                # Replicate the original file structure
                rel_path = os.path.relpath(mri_sub_folder, mri_folder)
                im = convert_file(mri_file_path)
                images.append(im)

    return images


def dicom_image_to_RGBA(image_data):
    rows = len(image_data)
    cols = rows
    img = np.empty((rows,cols), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((rows, cols, 4))
    for i in range(0,rows):
        for j in range(0,cols):
            view[i][j][0] = image_data[i][j]
            view[i][j][1] = image_data[i][j]
            view[i][j][2] = image_data[i][j]
            view[i][j][3] = 255
    return img