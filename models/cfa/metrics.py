from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def gaussian_smooth(x, sigma=4):
    bs = x.shape[0]
    for i in range(0, bs):
        x[i] = gaussian_filter(x[i], sigma=sigma)

    return x

def upsample(x, size, mode):
    return F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False).squeeze().numpy()