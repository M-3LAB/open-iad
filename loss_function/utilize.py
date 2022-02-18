import torch.nn.functional as F

__all__ = ['cal_loss']

def cal_loss(fs_list, ft_list, criterion):
    tot_loss = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        # a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        # f_loss = (1/(w*h))*torch.sum(a_map)
        f_loss = (0.5/(w*h))*criterion(fs_norm, ft_norm)
        tot_loss += f_loss

    return tot_loss