# FAVAE anomaly detection
This is an implementation of the paper [Anomaly localization by modeling perceptual features](https://arxiv.org/pdf/2008.05369.pdf)
<p align="center">
    <img src="imgs/pic1.jpg" width="600"\>
</p>

## Requirement
* python == 3.7
* pytorch == 1.5
* tqdm
* sklearn
* matplotlib

## Datasets
MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

## Code example
* bottle
```python
python train.py --obj bottle --do_aug
```

## Results

<p align="center">
    <img src="imgs/pic2.jpg" width="600"\>
</p>
<p align="center">
    <img src="imgs/pic3.jpg" width="600"\>
</p>
<p align="center">
    <img src="imgs/pic4.jpg" width="600"\>
</p>


## Reference
[1] David Dehaene, Pierre Eline. *Anomaly localization by modeling perceptual features*. https://arxiv.org/pdf/2008.05369.pdf

[2] https://github.com/byungjae89/SPADE-pytorch

[3] https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-
