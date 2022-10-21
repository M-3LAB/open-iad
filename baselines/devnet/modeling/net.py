import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        self.feature_extractor = build_feature_extractor(self.args.backbone)
        self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=1, kernel_size=1, padding=0)


    def forward(self, image):

        if self.args.n_scales == 0:
            raise ValueError

        image_pyramid = list()
        for s in range(self.args.n_scales):
            image_scaled = F.interpolate(image, size=self.args.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image_scaled)

            scores = self.conv(feature)
            if self.args.topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

            image_pyramid.append(scores)
        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)
        return score.view(-1, 1)
