# Anomaly-Detection-RGBD (ad-3d)
##Preliminary  

pytorch 1.10.1 or 1.8.1\
scikit-image, pip install scikit-image\
scikit-learn, pip instll scikit-learn\
opencv, pip install opencv-python

## Preliminary
> Dependency

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
## Fewshot AD
'''bash
python3 centralized_training.py --fewshot --fewshot-exm 5 --model 'patchcore2d' --chosen-train-task-ids 0 --chosen-test-task-id 1 -dg -g 0
'''bash
## Train MMAD
```bash
python3 mmad_training.py
```