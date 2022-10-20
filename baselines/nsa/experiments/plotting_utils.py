import torch 
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


def denorm(img):
  return img * torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1) + torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)


def show(img, ax=plt):
    npimg = img.cpu().numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)))


def plot_row(tensors, labels, ax, grid_cols=8):
  for i, (tensor, label) in enumerate(zip(tensors, labels)):
    if tensor is None: 
      ax[i].axis('off')
    else:
      tensor = tensor.cpu()
      if tensor.shape[1] == 3:
        tensor = denorm(tensor)
        normalize = False
      elif tensor.max() > 1 or tensor.min() < 0:
        normalize = True
      else:
        normalize = False
      img = make_grid(tensor, nrow=grid_cols, padding=2, normalize=normalize,
                  range=None, scale_each=False, pad_value=0)
      show(img, ax[i])
      ax[i].set_title(labels[i])
