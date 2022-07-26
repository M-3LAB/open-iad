# Anomaly-Detection-RGBD (ad-3d)
## Preliminary  

pytorch 1.10.1 or 1.8.1\
scikit-image, pip install scikit-image\
scikit-learn, pip instll scikit-learn\
opencv, pip install opencv-python

```bash
pip3 install -r requirements.txt
```
## Normal AD 
```bash
python3 centralized_training.py --model 'patchcore2d' --data-path '/disk2/mvtec/2D' 
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

## Fewshot AD
| Train | Test | Prototypes |
| ------ | -------|------ |
| all data (id=0) | all data (id=0) | normal |
| fewshot (id=0) | all data (id=0)  | fewshot-normal |
| all data (id=0) + fewshot (id=1) | all data (id=1) | fewshot |

--domain-generalization / -dg

> Normal
```bash
python3 centralized_training.py --model patchcore2d --chosen-train-task-ids 0 --chosen-test-task-id 0 -g 0
```

> Fewshot-Normal
```bash
python3 centralized_training.py --fewshot-normal --fewshot-exm 5 --model patchcore2d --chosen-train-task-ids 0 --chosen-test-task-id 0 --coreset-sampling-ratio 0.1 -dg -g 0
```

> Fewshot
```bash
python3 centralized_training.py --fewshot --fewshot-exm 5 --model patchcore2d --chosen-train-task-ids 0 --chosen-test-task-id 1 --coreset-sampling-ratio 0.1 -dg -g 0
```
