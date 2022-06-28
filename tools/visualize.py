import matplotlib.pyplot as plt
#from skimage import io
import cv2
import numpy as np

__all__ = ['show_cam_on_image', 'cv2heatmap', 'heatmap_on_image', 'min_max_norm',
           'cal_anomaly_map', 'save_anomaly_map']


def show_cam_on_image(img, anomaly_img):
    pass

def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image():
    pass

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min) 

def cal_anomaly_map():
    pass

def save_anomaly_map(anomaly_map, input_img, mask):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
    pass

if __name__ == 'main':
    pass