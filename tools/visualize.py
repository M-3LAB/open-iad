import cv2
import numpy as np
import os
from tools.utils import create_folders

__all__ = ['cv2heatmap', 'heatmap_on_image', 'min_max_norm', 'save_anomaly_map']


def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min) 


def save_anomaly_map(anomaly_map, input_img, mask, file_path):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    anomaly_map_norm = min_max_norm(anomaly_map) 
    heatmap = cv2heatmap(anomaly_map_norm * 255)

    heatmap_on_img = heatmap_on_image(heatmap, input_img)
    create_folders(file_path)

    cv2.imwrite(os.path.join(file_path, 'input.jpg'), input_img)
    cv2.imwrite(os.path.join(file_path, 'heatmap.jpg'), heatmap)
    cv2.imwrite(os.path.join(file_path, 'heatmap_on_img.jpg'), heatmap_on_img)
    cv2.imwrite(os.path.join(file_path, 'mask.jpg'), mask * 255)
