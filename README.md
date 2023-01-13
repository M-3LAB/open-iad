# Anomaly-Detection-RGBD (ad-3d)
## Preliminary  

pytorch 1.10.1 or 1.8.1\
scikit-image, pip install scikit-image\
scikit-learn, pip instll scikit-learn\
opencv, pip install opencv-python

```bash
pip3 install -r requirements.txt
```

## Project Instruction
```bash
├── arch_base # model base class
├── baselines # source code 
├── checkpoints # pretrained or requirements
├── configuration
│   ├── 1_model_base # highest priority
│   ├── 2_train_base # middle priority
│   ├── 3_dataset_base # lowest priority
│   ├── config.py # for main.py
│   └── device.py # for device
├── data_io # dataset processing and load data interface
├── legacy_code # old code, not used
├── loss_function
├── metrics
├── models # basic layers for model class in arch_base
├── optimizer
├── paradigms # learning paradigms
│   ├── centralized
│   │   ├── centralized_learning_2d.py # 2D
│   │   ├── centralized_learning_3d.py # 3D
│   └── federated
│       └── federated_learning_2d.py # 2D
├── run_scripts # shell code
├── tools
├── work_dir # save results
├── main.py # run start, with configuration/config.py
└── requirements.txt
```

## MVTec3D Preprocessing

> 3d_ast
```bash
cp -r ../zip/mvtec3d_official ./mvtec3d_ast
python3 ./baselines/3d_ast/preprocess.py
```
> 3d_btf
```bash
cp -r ../zip/mvtec3d_official ./mvtec3d_btf
python3 ./baselines/3d_btf/utils/preprocessing.py
```

## Dataset

--dataset / -d 

mvtec2d, mvtec3d, mpdd, mvtecloco, mtd, btad, mvtec2df3d, imad_hardware_parts


## Learning Paradigm

| Prototypes | Marker | Train | Test |
| ------ | ---| -------|------ |
| centralized 2d | -p c2d | |
| centralized 3d | -p c3d | |
| federated 2d | -p f2d | |
| vanilla | -v |all data (id=0) | all data (id=0) |
| semi | -s | all data (id=0) + anomaly data (id=0) | all data (id=0) - anomaly data (id=0)|
| continual | -c| all data (id=0 and 1)| all data (id=0 or 1)|
| fewshot | -f | fewshot (id=0) | all data (id=0) |
| noisy | -ny | all data (id=0) + noisy data (id=0) | all data (id=0) - noisy data (id=0)|

## Model
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


## Run Example

> Vanilla
```bash
python3 main.py -p c2d -v -m patchcore -n resnet18 -d mvtec2d -tid 0 -vid 0 --coreset-sampling-ratio 0.001 -g 1
python3 main.py -p c2d -v -m csflow -n net_csflow -d mvtec2d -tid 11 -vid 11 -g 1
python3 main.py -p c2d -v -m cfa -n net_cfa -d mvtec2d -tid 11 -vid 11 --coreset-sampling-ratio 0.001 -g 7
python3 main.py -p c2d -v -m draem -n net_draem -d mvtec2d -tid 11 -vid 11 -g 2
python3 main.py -p c2d -v -m fastflow -n net_fastflow -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -v -m cutpaste -n vit_b_16  -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -v -m padim -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -v -m favae -n net_favae -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -v -m cutpaste -n vit_b_16 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -v -m igd -n net_igd -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -v -m reverse -n net_reverse -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -v -m spade -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -v -m stpm -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7

```

> Continual
```bash
python3 main.py -p c2d -c -m patchcore -n resent18 -d mvtec2d -tid 0 1 -vid 0 1 --coreset-sampling-ratio 0.001 -g 1
python3 main.py -p c2d -c -m csflow -n net_csflow -d mvtec2d -tid 10 11 -vid 10 11 -g 1
python3 main.py -p c2d -c -m cfa -n net_cfa -d mvtec2d -tid 10 11 -vid 10 11 --coreset-sampling-ratio 0.001 -g 7
python3 main.py -p c2d -c -m draem -n net_draem -d mvtec2d -tid 10 11 -vid 10 11 -g 2
python3 main.py -p c2d -c -m fastflow -n net_fastflow -d mvtec2d -tid 10 11 -vid 10 11 -g 7
python3 main.py -p c2d -c -m cutpaste -n vit_b_16  -d mvtec2d -tid 10 11 -vid 10 11  -g 7
python3 main.py -p c2d -c -m padim -n resnet18 -d mvtec2d -tid 10 11 -vid 10 11 -g 7
python3 main.py -p c2d -c -m favae -n net_favae -d mvtec2d -tid 10 11 -vid 10 11  -g 7
python3 main.py -p c2d -c -m cutpaste -n vit_b_16 -d mvtec2d -tid 10 11 -vid 10 11 -g 7
python3 main.py -p c2d -c -m igd -n net_igd -d mvtec2d -tid 10 11 -vid 10 11 -g 7
python3 main.py -p c2d -c -m reverse -n net_reverse -d mvtec2d -tid 10 11 -vid 10 11 -g 7
python3 main.py -p c2d -c -m spade -n resnet18 -d mvtec2d -tid 10 11 -vid 10 11 -g 7
python3 main.py -p c2d -c -m stpm -n resnet18 -d mvtec2d -tid 10 11 -vid 10 11 -g 7
python3 main.py -p c2d -c -m dne -n vit_b_16 -d mvtec2d -tid 10 11 -vid 10 11 -g 7
```

> Fewshot
```bash
python3 mian.py -p c2d -f --fewshot-exm 1 --fewshot-num-dg 4 -m patchcore -n resnet18 -d mvtec2d -tid 0 -vid 0 --coreset-sampling-ratio 1 -g 1
python3 mian.py -p c2d -f --fewshot-exm 1 -m patchcore -n resnet18 -d mvtec2d -tid 0 -vid 0 --coreset-sampling-ratio 1 -g 1
python3 mian.py -p c2d -f --fewshot-exm 1 -m csflow -n net_csflow -d mvtec2d -tid 11 -vid 11 -g 1
python3 mian.py -p c2d -f --fewshot-exm 1 -m cfa -n net_cfa -d mvtec2d -tid 11 -vid 11 --coreset-sampling-ratio 1 -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m draem -n net_draem -d mvtec2d -tid 11 -vid 11 -g 2
python3 mian.py -p c2d -f --fewshot-exm 1 -m fastflow -n net_fastflow -d mvtec2d -tid 11 -vid 11 -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m cutpaste -n vit_b_16  -d mvtec2d -tid 11 -vid 11  -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m padim -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m favae -n net_favae -d mvtec2d -tid 11 -vid 11  -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m cutpaste -n vit_b_16 -d mvtec2d -tid 11 -vid 11 -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m igd -n net_igd -d mvtec2d -tid 11 -vid 11 -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m reverse -n net_reverse -d mvtec2d -tid 11 -vid 11 -g 7
python3 mian.py -p c2d -f --fewshot-exm 8 -m spade -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 mian.py -p c2d -f --fewshot-exm 1 -m stpm -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
```
> Semi
```bash
python3 mian.py -p c2d -s -m devnet -n net_devnet -d mvtec2d -tid 0 -vid 0 -g 1
python3 mian.py -p c2d -s -m dra -n net_dra -d mvtecloco -tid 0 -vid 0 -g 1
```

> Noisy
```bash
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m patchcore -n resnet18 -d mvtec2d -tid 0 -vid 1 --coreset-sampling-ratio 0.001 -g 1
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m csflow -n net_csflow -d mvtec2d -tid 11 -vid 11 -g 1
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m cfa -n net_cfa -d mvtec2d -tid 11 -vid 11 --coreset-sampling-ratio 0.001 -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m draem -n net_draem -d mvtec2d -tid 11 -vid 11 -g 2
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m fastflow -n net_fastflow -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m cutpaste -n vit_b_16  -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m padim -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m favae -n net_favae -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m cutpaste -n vit_b_16 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m reverse -n net_reverse -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m spade -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m stpm -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -ny --noisy-ratio 0.1 --noisy-overlap -m igd -n net_igd -d mvtec2d -tid 11 -vid 11 -g 7
```


