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
python3 data_io/preprocessing.py --dataset-path '/disk/mvtec/3D'
```

## Train MMAD
```bash
python3 mmad_training.py
```

## Learning Paradigm
| Prototypes | Train | Test |
| ------ | -------|------ |
| vanilla |all data (id=0) | all data (id=0) 
| continual | all data (id=0 and 1)| all data (id=0 or 1)|
| fewshot | fewshot (id=0) | all data (id=0) |
| noisy | all data (id=0) + noisy data (id=0) | all data (id=0)|



> Vanilla
```bash
python3 centralized_training.py --vanilla --model patchcore2d --dataset mvtec2d --train-task-id 0 --valid-task-id 0 --coreset-sampling-ratio 0.001 -g 1
```

> Continual
```bash
python3 centralized_training.py --continual --model patchcore2d --dataset mvtec2d --train-task-id 0 1 --valid-task-id 0 1 --coreset-sampling-ratio 0.001 -g 1
```

> Fewshot
```bash
python3 centralized_training.py --fewshot --fewshot-exm 1 --fewshot-num-dg 4 --model patchcore2d --dataset mvtec2d --train-task-id 0 --valid-task-id 0 --coreset-sampling-ratio 1 -g 1
```

> Noisy
```bash
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model patchcore2d --dataset mvtec2d --train-task-id 0 --valid-task-id 1 --coreset-sampling-ratio 0.001 -g 1
```
