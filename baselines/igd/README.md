# IGD
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-anomaly-detection-and/anomaly-detection-on-mnist)](https://paperswithcode.com/sota/anomaly-detection-on-mnist?p=unsupervised-anomaly-detection-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-anomaly-detection-and/anomaly-detection-on-fashion-mnist)](https://paperswithcode.com/sota/anomaly-detection-on-fashion-mnist?p=unsupervised-anomaly-detection-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-anomaly-detection-and/anomaly-detection-on-one-class-cifar-10)](https://paperswithcode.com/sota/anomaly-detection-on-one-class-cifar-10?p=unsupervised-anomaly-detection-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-anomaly-detection-and/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=unsupervised-anomaly-detection-and)

This repo contains the Pytorch implementation of our paper:
> [**Deep One-Class Classification via Interpolated Gaussian Descriptor**](https://arxiv.org/pdf/2101.10043.pdf)
>
> Yuanhong Chen*, [Yu Tian*](https://yutianyt.com/), [Guansong Pang](https://sites.google.com/site/gspangsite/home?authuser=0), [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **Accepted at AAAI 2022 (Oral).**  

## Dataset

[**Please download the MVTec AD dataset**](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## Train and Test IGD 
After the setup, simply run the following command to train/test the global/local model: 
```shell
./job.sh
```


## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{chen2022deep,
  title={Deep one-class classification via interpolated gaussian descriptor},
  author={Chen, Yuanhong and Tian, Yu and Pang, Guansong and Carneiro, Gustavo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={1},
  pages={383--392},
  year={2022}
}
```
---
