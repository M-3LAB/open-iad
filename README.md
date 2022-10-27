# Anomaly-Detection-RGBD (ad-3d)
## Preliminary  

pytorch 1.10.1 or 1.8.1\
scikit-image, pip install scikit-image\
scikit-learn, pip instll scikit-learn\
opencv, pip install opencv-python

```bash
pip3 install -r requirements.txt
```

## Install Third Party Library
```bash
bash setup.sh
```

## MVTec3D Preprocessing (Denoise Data)
```bash
python3 data_io/preprocessing.py --dataset-path '/disk/mvtec3d'
```

## Train MMAD
```bash
python3 mmad_training.py
```

## Learning Paradigm
| Prototypes | Marker | Train | Test |
| ------ | ---| -------|------ |
| vanilla | -v |all data (id=0) | all data (id=0) |
| semi | -s | all data (id=0) + anomaly data (id=0) | all data (id=0) - anomaly data (id=0)|
| continual | -c| all data (id=0 and 1)| all data (id=0 or 1)|
| fewshot | -f | fewshot (id=0) | all data (id=0) |
| noisy | -n | all data (id=0) + noisy data (id=0) | all data (id=0) - noisy data (id=0)|


| Method / -m | Net / -n |
| ------ | ------ |
| patchcore  | resnet18, wide_resnet50 |
| padim  | resnet18, wide_resnet50 |
| csflow | net_csflow |
| dne | vit_b_16 |
| draem | net_draem |
| dra | net_dra |
| igd | net_igd |



> Vanilla
```bash
python3 centralized_training.py --vanilla --model patchcore --net resent18 --dataset mvtec2d --train-task-id 0 --valid-task-id 0 --coreset-sampling-ratio 0.001 -g 1
```

> Continual
```bash
python3 centralized_training.py --continual --model patchcore --net resent18 --dataset mvtec2d --train-task-id 0 1 --valid-task-id 0 1 --coreset-sampling-ratio 0.001 -g 1
```

> Fewshot
```bash
python3 centralized_training.py --fewshot --fewshot-exm 1 --fewshot-num-dg 4 --model patchcore --net resent18 --dataset mvtec2d --train-task-id 0 --valid-task-id 0 --coreset-sampling-ratio 1 -g 1
```

> Noisy
```bash
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model patchcore --net resent18 --dataset mvtec2d --train-task-id 0 --valid-task-id 1 --coreset-sampling-ratio 0.001 -g 1
```


