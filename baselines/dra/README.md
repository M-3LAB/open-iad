# Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection (CVPR2022)
By Choubo Ding, Guansong Pang, Chunhua Shen

Official PyTorch implementation of ["Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection"](https://arxiv.org/abs/2203.14506).

## Prerequisites 
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

## Run

#### Step 1. Setup the Anomaly Detection Dataset
Download the Anomaly Detection Dataset and convert it to MVTec AD format. (For datasets we used in the paper, we provided the [convert script](https://github.com/Choubo/DRA/tree/main/data).) 
The dataset folder structure should look like:
```
DATA_PATH/
    subset_1/
        train/
            good/
        test/
            good/
            defect_class_1/
            defect_class_2/
            defect_class_3/
            ...
    ...
```

#### Step 2. Running DRA
```bash
python train.py --dataset_root=./data/mvtec_anomaly_detection \
                --classname=carpet \
                --experiment_dir=./experiment
```
- `dataset_root` denotes the path of the dataset.
- `classname` denotes the subset name of the dataset.
- `experiment_dir` denotes the path to store the experiment setting and model weight.
- `outlier_root` (*optional) given the path of the outlier dataset to disable pseudo augmentation and enable external data for pseudo head.
- `know_class` (*optional) specify the anomaly class in the training set to experiment within the hard setting.

## Citation
```bibtex
@inproceedings{ding2022catching,
      title={Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection}, 
      author={Ding, Choubo and Pang, Guansong and Shen, Chunhua},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2022}
}
```