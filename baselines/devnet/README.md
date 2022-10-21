# Explainable Deep Few-shot Anomaly Detection with Deviation Networks
By Guansong Pang, Choubo Ding, Chunhua Shen, Anton van den Hengel

Official PyTorch implementation of ["Explainable Deep Few-shot Anomaly Detection with Deviation Networks"](https://arxiv.org/abs/2108.00462).
## Setup 
This code is written in `Python 3.6` and requires the packages listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

## Usage

#### Step 1. Setup the Anomaly Detection Dataset
Download the Anomaly Detection Dataset and convert it to MVTec AD format. (For datasets we used in the paper, we provided the convert script.) 
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
        ground_truth/
            defect_class_1/
            defect_class_2/
            defect_class_3/
            ...
    ...
```
NOTE: The `ground_truth` folder only available when the dataset has pixel-level annotation.

#### Step 2. Running DevNet
```bash
python train.py --dataset_root=./data/mvtec_anomaly_detection \
                --classname=carpet \
                --experiment_dir=./experiment \
                --epochs=50 \
                --n_anomaly=10 \
                --n_scales=2
```
- `dataset_root` denotes the path of the dataset.
- `classname` denotes the subset name of the dataset.
- `experiment_dir` denotes the path to store the experiment setting and model weight.
- `epochs` denotes the total epoch of training. 
- `n_anomaly` denotes the amount of the know outliers. 
- `n_scales` denotes the total scales of multi-scales module. 

#### Step 2. Anomaly Explanation
Visualize the localization result of the trained model by the following command:
```bash
python localization.py --dataset_root=./data/mvtec_anomaly_detection \
                       --classname=carpet \
                       --experiment_dir=./experiment \
                       --n_anomaly=10 \
                       --n_scales=2
```
NOTE: use same argument as the training command.
## Citation
```bibtex
@article{pang2021explainable,
  title={Explainable Deep Few-shot Anomaly Detection with Deviation Networks},
  author={Pang, Guansong and Ding, Choubo and Shen, Chunhua and Hengel, Anton van den},
  journal={arXiv preprint arXiv:2108.00462},
  year={2021}
}
```
