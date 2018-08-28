from tkinter import *
import DicomHandler as DH
import LungSegmentation as LS
import numpy as np
from tkinter import filedialog
from scipy.ndimage import morphology
import neural_network as nn
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import os

loaded = False
switched_on = True
png_images = []
pngs = []
percentages = []
index = 0

def generate_labels(dir, png_images):
    non_labeled_imgs = LS.load_scan(dir)
    hu_value_imgs = LS.get_pixels_hu(non_labeled_imgs)[::-1]
    masks = LS.segment_lung_mask(hu_value_imgs)
    binary_masks = morphology.binary_fill_holes(
        morphology.binary_dilation(
            morphology.binary_fill_holes(masks > 0),
            iterations=1)
    )
    segmented_lungs = hu_value_imgs * binary_masks
    middle_point = int(21 / 2)
    label_list = []
    for k in range(len(segmented_lungs)):
        segmented_image = segmented_lungs[k]
        mask = binary_masks[k]
        label_matrix = np.zeros((512, 512), dtype=np.int)
        if 1 in mask:
            i = 0
            for i in range(0, segmented_image.shape[1], 7):
                j = 0
                while j < segmented_image.shape[0]:
                    if mask[i][j] != 0:
                        cropped_image = segmented_image[i - middle_point: i + middle_point + 1,
                                        j - middle_point: j + middle_point + 1]
                        label = nn.test_frame(cropped_image) + 1
                        label_matrix[i - 4: i + 3, j - 4 : j + 3] = label
                        j = j + 7
                    else:
                        j += 1
        label_list.append(label_matrix)

    im_list = []
    percentage_list = []
    for k in range(len(binary_masks)):
        mask = binary_masks[k]
        percentage = np.zeros(3)
        if 1 in mask:
            label_matrix = label_list[k]
            im = cv2.cvtColor(png_images[k], cv2.COLOR_GRAY2RGB)
            for i in range(0, mask.shape[1]):
                for j in range(0, mask.shape[1]):
                    if label_matrix[i][j] != 0 and mask[i][j] != 0:
                        if label_matrix[i][j] == 1:
                            im[i, j] = [255, 0, 0]
                            percentage[0] += 1
                        elif label_matrix[i][j] == 2:
                            im[i, j] = [0, 255, 0]
                            percentage[1] += 1
                        elif label_matrix[i][j] == 3:
                            im[i, j] = [0, 0, 255]
                            percentage[2] += 1
            im_list.append(im)
            percentage_list.append(percentage)
        else:
            im_list.append(cv2.cvtColor(png_images[k], cv2.COLOR_GRAY2RGB))
            percentage_list.append(percentage)
    return im_list, percentage_list


def folder_selector_button_clicked(panel, slice_index_button):
    global png_images
    global loaded
    global pngs, index, percentages
    loaded = False
    dir = filedialog.askdirectory()
    dir = os.path.abspath(dir)
    pngs = DH.convert_folder(dir)[::-1]

    png_images, percentages = generate_labels(dir, pngs)
    loaded = True
    index = 0
    img = ImageTk.PhotoImage(Image.fromarray(np.uint8(png_images[0]), 'RGB'))
    panel.configure(image=img)
    panel.image = img  # keep a reference!
    percentage = percentages[0]
    slice_index_button.configure(text="Slice:{}, Honeycomb:{:.0%}, GroundGlass:{:.0%}, Healthy:{:.0%}".
                                 format(index, percentage[0] / np.sum(percentage), percentage[1] / np.sum(percentage),
                                        percentage[2] / np.sum(percentage)))
    #slice_index_label.config(text="Slice : 0")


def next(panel, slice_index_button):
    global index
    if loaded:
        index = index + 1
        if index >= len(png_images):
            index = 0
        img = ImageTk.PhotoImage(Image.fromarray(np.uint8(png_images[index]), 'RGB'))
        panel.configure(image=img)
        panel.image = img # keep a reference!
        percentage = percentages[index]
        slice_index_button.configure(text="Slice:{}, Honeycomb:{:.0%}, GroundGlass:{:.0%}, Healthy:{:.0%}".
                                     format(index, percentage[0] / np.sum(percentage),
                                            percentage[1] / np.sum(percentage),
                                            percentage[2] / np.sum(percentage)))
        #slice_index_label.config(text="Slice : " + str(index))


def prev(panel, slice_index_button):
    global index
    if loaded:
        index = index - 1
        if index < 0:
            index = len(png_images) - 1
        img = ImageTk.PhotoImage(Image.fromarray(np.uint8(png_images[index]), 'RGB'))
        panel.configure(image=img)
        panel.image = img # keep a reference!
        percentage = percentages[index]
        slice_index_button.configure(text="Slice:{}, Honeycomb:{:.0%}, GroundGlass:{:.0%}, Healthy:{:.0%}".
                                     format(index, percentage[0] / np.sum(percentage),
                                            percentage[1] / np.sum(percentage),
                                            percentage[2] / np.sum(percentage)))
        #slice_index_label.config(text="Slice : " + str(index))


def switch_labels(panel):
    global index, switched_on
    if switched_on:
        img = ImageTk.PhotoImage(Image.fromarray(pngs[index]))
        panel.configure(image=img)
        panel.image = img
        switched_on = False
    else:
        img = ImageTk.PhotoImage(Image.fromarray(np.uint8(png_images[index]), 'RGB'))
        panel.configure(image=img)
        panel.image = img
        switched_on = True

#Create main window
window = tk.Tk()
window.title("Honeycomb & Ground Glass Area Detection")
window.geometry('1024x720')

#divide window into two sections. One for image. One for buttons
top = tk.Frame(window)
top.pack(side="top")
bottom = tk.Frame(window)
bottom.pack(side="bottom")

#place image
logo = os.path.join(os.getcwd(), 'resources')
img = ImageTk.PhotoImage(Image.open(os.path.join(logo, "logo.png")))
panel = tk.Label(window,image=img, width=512, height=512)
panel.image = img  # keep a reference!
panel.pack(side = "top")



#place buttons
slice_index_button = tk.Button(window, text="Slice:0, Honeycomb:%0, GroundGlass:%0, Healthy:%0", width=50, height=100)
slice_index_button.pack(side="left")
prev_button = tk.Button(window, text="Previous", width=10, height=2, borderwidth=5, command=lambda: prev(panel, slice_index_button))
prev_button.pack(in_=bottom, side="left", pady = 30)
next_button = tk.Button(window, text="Next", width=10, height=2, borderwidth=5, command=lambda: next(panel, slice_index_button))
next_button.pack(in_=bottom, side="right", pady = 30)
switch_labels_button = tk.Button(window, text="Turn On/Off Labels", width=20, height=2,  borderwidth=5, command=lambda: switch_labels(panel))
switch_labels_button.pack(in_=top, side="right", pady = 10)



folder_selector_button = tk.Button(window, text="Select Folder", width=10, height=2, borderwidth=5, command=lambda: folder_selector_button_clicked(panel, slice_index_button))
folder_selector_button.pack(in_=top, side="top", pady = 10)
#Start the GUI
window.mainloop()


