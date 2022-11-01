import matplotlib.pyplot as plt
#from skimage import io
import cv2
import numpy as np
import os
from tools.utils import create_folders
from sklearn.manifold import TSNE
from time import time

__all__ = ['show_cam_on_image', 'cv2heatmap', 'heatmap_on_image', 'min_max_norm',
           'cal_anomaly_map', 'save_anomaly_map', 'plot_embedding', 'vis_embedding']


def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

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

def cal_anomaly_map():
    pass

def save_anomaly_map(anomaly_map, input_img, mask, file_path):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    anomaly_map_norm = min_max_norm(anomaly_map) 
    heatmap = cv2heatmap(anomaly_map_norm*255)

    heatmap_on_img = heatmap_on_image(heatmap, input_img)
    create_folders(file_path)

    cv2.imwrite(os.path.join(file_path, 'input.jpg'), input_img)
    cv2.imwrite(os.path.join(file_path, 'heatmap.jpg'), heatmap)
    cv2.imwrite(os.path.join(file_path, 'heatmap_on_img.jpg'), heatmap_on_img)
    cv2.imwrite(os.path.join(file_path, 'mask.jpg'), mask)
    
def plot_embedding(data, label, num_shot, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    p1 = None
    p2 = None
    for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
        if label[i] < num_shot:
            p1 = plt.scatter(data[i, 0], data[i, 1], lw=1, color=plt.cm.Set1(label[i]), marker='.')
        else:
            p2 = plt.scatter(data[i, 0], data[i, 1], lw=1, color=plt.cm.Set1(label[i]), marker='x')
    plt.xticks([])
    plt.yticks([])
    plt.legend([p1, p2], ['fewshot', 'dg'])
    plt.title(title)

    return fig

def vis_embeddings(data, label, num_shot, file_path):
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label, num_shot, 't-SNE (time %.2fs)' % (time() - t0))
    # plt.show(fig)
    plt.savefig(file_path)


if __name__ == 'main':
    pass