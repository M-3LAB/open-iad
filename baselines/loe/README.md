# Latent Outlier Exposure for Anomaly Detection with Contaminated Data (LOE)

This is the companion code for a PyTorch implementation of Latent Outlier Exposure reported in the paper
**Latent Outlier Exposure for Anomaly Detection with Contaminated Data** by Chen Qiu et al. 
The paper is published in ICML 2022 and can be found here https://arxiv.org/abs/2202.08088. 
The code allows the users to reproduce and extend the results reported in the study. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## Reproduce the Results

This repo contains the code of experiments with LOE on various data types including image data and tabular data. The implementation of the backbone anomaly detector (Neural Transformation Learning) is based on the code from https://github.com/boschresearch/NeuTraL-AD.

Please run the command and replace \$# with available options (see below): 

```
python Launch_Exps.py --config-file $1 --dataset-name $2  --contamination $3
```

**config-file:** 

* config_cifar10.yml; config_fmnist.yml; config_thyroid.yml; config_arrhy.yml; 

**dataset-name:** 

* cifar10 (image); fmnist (image); thyroid (tabular); arrhythmia (tabular);

**contamination:** 

+ The ground-truth contamination ratio of the dataset. The default ratio is 0.1.


## How to Use
1. When using your own data, please put your data files under [DATA](DATA).

2. Create a config file which contains your hyper-parameters under [config_files](config_files).  

3. Add your data loader to the function ''load_data'' in the [loader/LoadData.py](loader/LoadData.py).
* The shape is (batch size, feature dim).

## Datasets

* Arrhythmia and Thyroid datasets are downloaded from https://github.com/lironber/GOAD. Please put the data under [DATA](DATA).  

* Cifar10/fmnist is the last-layer features of Cifar 10/FashionMNIST extracted by a ResNet152 pretrained on ImageNet. [Extract_img_features.py](Extract_img_features.py) is used to extract features.
## License

Latent Outlier Exposure for Anomaly Detection with Contaminated Data (LOE) is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Latent Outlier Exposure for Anomaly Detection with Contaminated Data (LOE) , see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
