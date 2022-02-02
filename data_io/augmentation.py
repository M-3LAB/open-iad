from torchvision import transform as T

__all__ =  ['standar_image_transform', 'standard_mask_transform']

standard_image_transform = T.Compose([T.Resize(224),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                                      ])

standard_mask_transform = T.Compose([T.Resize(224),
                                     T.CenterCrop(224),
                                     T.ToTensor()
                                     ])