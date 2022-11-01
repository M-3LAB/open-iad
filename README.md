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
| noisy | -ny | all data (id=0) + noisy data (id=0) | all data (id=0) - noisy data (id=0)|


| Method / -m | Net / -n |
| ------ | ------ |
| cfa | net_cfa |
| csflow | net_csflow |
| cutpaste | vit_b_16 |
| devnet | net_devnet |
| dne | vit_b_16 |
| dra | net_dra |
| draem | net_draem |
| fastflow | net_fastflow |
| favae | net_favae |
| igd | net_igd |
| padim  | resnet18, wide_resnet50 |
| patchcore  | resnet18, wide_resnet50 |
| reverse | net_reverse |
| spade  | resnet18, wide_resnet50 |
| stpm  | resnet18, wide_resnet50 |



> Vanilla
```bash
python3 centralized_training.py --vanilla --model patchcore --net resnet18 --dataset mvtec2d --train-task-id 0 --valid-task-id 0 --coreset-sampling-ratio 0.001 -g 1
python3 centralized_training.py --vanilla --model csflow --net net_csflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 1
python3 centralized_training.py --vanilla --model cfa --net net_cfa --dataset mvtec2d --train-task-id 11 --valid-task-id 11 --coreset-sampling-ratio 0.001 -g 7
python3 centralized_training.py -v --model draem --net net_draem --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 2
python3 centralized_training.py -v --model fastflow -n net_fastflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py -v --model cutpaste -n vit_b_16  --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py -v --model padim -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py -v --model favae --net net_favae --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py -v --model cutpaste -n vit_b_16 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py -v --model igd -n net_igd --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py -v --model reverse -n net_reverse --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py -v --model spade -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py -v --model stpm -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7

```

> Continual
```bash
python3 centralized_training.py --continual --model patchcore --net resent18 --dataset mvtec2d --train-task-id 0 1 --valid-task-id 0 1 --coreset-sampling-ratio 0.001 -g 1
python3 centralized_training.py --continual --model csflow --net net_csflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 1
python3 centralized_training.py --continual --model cfa --net net_cfa --dataset mvtec2d --train-task-id 11 --valid-task-id 11 --coreset-sampling-ratio 0.001 -g 7
python3 centralized_training.py --continual --model draem --net net_draem --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 2
python3 centralized_training.py --continual --model fastflow -n net_fastflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --continual --model cutpaste -n vit_b_16  --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py --continual --model padim -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --continual --model favae --net net_favae --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py --continual --model cutpaste -n vit_b_16 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --continual --model igd -n net_igd --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --continual --model reverse -n net_reverse --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --continual --model spade -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --continual --model stpm -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py -c --model dne --net vit_b_16 --dataset mvtec2d --train-task-id 0 11 --valid-task-id 0 11 -g 7
```

> Fewshot
```bash
python3 centralized_training.py --fewshot --fewshot-exm 1 --fewshot-num-dg 4 --model patchcore --net resnet18 --dataset mvtec2d --train-task-id 0 --valid-task-id 0 --coreset-sampling-ratio 1 -g 1
python3 centralized_training.py --fewshot --fewshot-exm 1 --model patchcore --net resnet18 --dataset mvtec2d --train-task-id 0 --valid-task-id 0 --coreset-sampling-ratio 1 -g 1
python3 centralized_training.py --fewshot --fewshot-exm 1 --model csflow --net net_csflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 1
python3 centralized_training.py --fewshot --fewshot-exm 1 --model cfa --net net_cfa --dataset mvtec2d --train-task-id 11 --valid-task-id 11 --coreset-sampling-ratio 1 -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model draem --net net_draem --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 2
python3 centralized_training.py --fewshot --fewshot-exm 1 --model fastflow -n net_fastflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model cutpaste -n vit_b_16  --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model padim -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model favae --net net_favae --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model cutpaste -n vit_b_16 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model igd -n net_igd --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model reverse -n net_reverse --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --fewshot --fewshot-exm 8 --model spade -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --fewshot --fewshot-exm 1 --model stpm -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
```
> Semi
```bash
python3 centralized_training.py -s --model devnet --net net_devnet --dataset mvtec2d --train-task-id 0 --valid-task-id 0 -g 1
python3 centralized_training.py -s --model dra --net net_dra --dataset mvtecloco --train-task-id 0 --valid-task-id 0 -g 1
```

> Noisy
```bash
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model patchcore --net resnet18 --dataset mvtec2d --train-task-id 0 --valid-task-id 1 --coreset-sampling-ratio 0.001 -g 1
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model csflow --net net_csflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 1
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model cfa --net net_cfa --dataset mvtec2d --train-task-id 11 --valid-task-id 11 --coreset-sampling-ratio 0.001 -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model draem --net net_draem --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 2
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model fastflow -n net_fastflow --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model cutpaste -n vit_b_16  --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model padim -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model favae --net net_favae --dataset mvtec2d --train-task-id 11 --valid-task-id 11  -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model cutpaste -n vit_b_16 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model igd -n net_igd --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model reverse -n net_reverse --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model spade -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
python3 centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model stpm -n resnet18 --dataset mvtec2d --train-task-id 11 --valid-task-id 11 -g 7
```


