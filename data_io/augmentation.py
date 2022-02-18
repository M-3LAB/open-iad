from torchvision import transforms as T

__all__ =  ['mvtec_2d_image_transform', 'mvtec_2d_mask_transform']

mvtec_2d_image_transform = T.Compose([T.Resize(224),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                                      ])

mvtec_2d_mask_transform = T.Compose([T.Resize(224),
                                     T.CenterCrop(224),
                                     T.ToTensor()
                                     ])

#TODO: 3D has not been finished yet