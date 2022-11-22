import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# x = torch.randn(1, 3, 224, 224)
# # output = torch.nn.functional.unfold(x, kernel_size=(2, 3))
# print(x)
# print(x.size())
# x1 = x[:,:,0::4,0::4]
# x2 = x[:,:,1::4,1::4]
# x3 = x[:,:,2::4,2::4]
# x4 = x[:,:,3::4,3::4]
# print(x1.size())
# print(x2.size())
# print(x3.size())
# print(x4.size())

# output = torch.cat([x1,x2,x3,x4],dim=1)
# output = torch.nn.functional.conv2d(x,kernel_size=4,stride=4)
# print(output.size())

class Model_d(torch.nn.Module):
    def __init__(self):
        super(Model_d, self).__init__()
        self.convl = torch.nn.Sequential(
            torch.nn.Conv2d(12, 64, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU()
        )
    def forward(self, x):
        x1 = x[:,:,0::4,0::4]
        x2 = x[:,:,1::4,1::4]
        x3 = x[:,:,2::4,2::4]
        x4 = x[:,:,3::4,3::4]
        x =  torch.cat([x1,x2,x3,x4],dim=1)
        x = self.convl(x)
        return x
