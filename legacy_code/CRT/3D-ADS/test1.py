import torch
from data.mvtec3d import get_data_loader
from tqdm import tqdm

class runner():
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.methods = {}
        self.avgpooling = torch.nn.AvgPool2d(kernel_size=2,stride=2)

    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size)
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            train_data_D = sample[2]
            print(train_data_D.size())
            x = self.avgpooling(train_data_D)
            print(x.size())


    # def evaluate(self, class_name):
    #     image_rocaucs = dict()
    #     pixel_rocaucs = dict()
    #     au_pros = dict()
    #     test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size)
    #     with torch.no_grad():
    #         for sample, mask, label in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
    #             for method in self.methods.values():
    #                 method.predict(sample, mask, label)

    #     for method_name, method in self.methods.items():
    #         method.calculate_metrics()
    #         image_rocaucs[method_name] = round(method.image_rocauc, 3)
    #         pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
    #         au_pros[method_name] = round(method.au_pro, 3)
    #         print(
    #             f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
    #     return image_rocaucs, pixel_rocaucs, au_pros







# x = torch.randn(2, 5, 4, 5)
# output = torch.nn.functional.unfold(x, kernel_size=(2, 3))
# # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
# # 4 blocks (2x3 kernels) in total in the 3x4 input
# print(output.size())
# # torch.Size([2, 30, 4])