# IM-IAD: Industrial Image Anomaly Detection Benchmark in Manufacturing 


We are dedicated to provide researchers a uniform verification environment of image anomaly detection with standard settings and methods. At the same time, everyone is warmly invited to integrate their algorithms into our project, continuously complementing IM-IAD.
Let's together maintain and contribute this community.

## Envs

```bash
pytorch 1.10.1 or 1.8.1
conda activate open-iad
pip3 install -r requirements.txt

# example
pip install scikit-image
pip instll scikit-learn
pip install opencv-python
```

## Project Instruction
```bash
.
├── arch # model base class
├── augmentation # data augmentation
├── checkpoints # pretrained or requirements
├── configuration
│   ├── 1_model_base # highest priority
│   ├── 2_train_base # middle priority
│   ├── 3_dataset_base # lowest priority
│   ├── config.py # for main.py
│   ├── device.py # for device
│   └── registeration.py # register new model, dataset, server
├── data_io # loading data interface
├── dataset # dataset interface
├── loss_function
├── metrics
├── models # basic layers for model class in arch_base
├── optimizer
├── paradigms # learning paradigms
│   ├── centralized
│   │   ├── c2d.py # 2D
│   │   ├── c3d.py # 3D
│   └── federated
│       └── f2d.py # 2D
├── tools
├── work_dir # save results
├── main.py # run start, with configuration/config.py
└── requirements.txt
```

## Dataset (--dataset / -d)

> 2D: mvtec2d, mpdd, mvtecloco, mtd, btad, mvtec2df3d, coad

> 3D: mvtec3d

The dataset's structure can be organized as follows (i.e., mvtec2d).
```bash
.
├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   │   ├── 000_mask.png
│   │   │   ├── 001_mask.png
│   │   │   ├── ...
│   ├── test
│   │   ├── broken_large
│   │   │   ├── 000.png
│   │   │   ├── 001.png
│   │   │   ├── ...
│   │   └── good
│   │       ├── 000.png
│   │       ├── 001.png
│   │       ├── ...
│   └── train
│       └── good
│           ├── 000.png
│           ├── 001.png
│           ├── ...
├── cable
├── screw
└── ...
```
## Learning Paradigm

|| Prototypes | Marker | Train | Test |
| ------ | ------ | ---| -------|------ |
| $\bigstar$ | *centralized 2d* | -p c2d | |
|  | vanilla | -v |all data (id=0) | all data (id=0) |
|  | semi | -s | all data (id=0) + anomaly data (id=0) | all data (id=0) - anomaly data (id=0)|
|  | fewshot | -f | fewshot (id=0) | all data (id=0) |
|  | continual | -c| all data (id=0 and 1)| all data (id=0 or 1)|
|  | noisy | -z | all data (id=0) + noisy data (id=0) | all data (id=0) - noisy data (id=0)|
|  | transfer | -t | step 1: all data (id=0) | all data (id=0)|
|  |  |  | step 2: fewshot data (id=1) | all data (id=1)|
| $\bigstar$ | *centralized 3d* | -p c3d | To be updated! |
| $\bigstar$ | *federated 2d* | -p f2d |  To be updated! |

## 2D Model
| No. | Method / -m | Net / -n | Paper Title|
| ------ | ------ | ------ | ------ |
| 1 | cfa | net_cfa | CFA: Coupled-hypersphere-based feature adaptation for target-oriented anomaly localization |
| 2 | csflow | net_csflow | Fully convolutional cross-scale-flows for image-based defect detection |
| 3 | cutpaste | vit_b_16 | Cutpaste: self-supervised learning for anomaly detection and localization |
| 4 | devnet | net_devnet | Explainable deep few-shot anomaly detection with deviation networks |
| 5 | dne | vit_b_16 | Towards continual adaptation in industrial anomaly detection |
| 6 | dra | net_dra | Catching both gray and black swans: open-set supervised anomaly detection |
| 7 | draem | net_draem | Draem: a discriminatively trained reconstruction embedding for surface anomaly detection |
| 8 | fastflow | net_fastflow | Fastflow: unsupervised anomaly detection and localization via 2d normalizing flows |
| 9 | favae | net_favae | Anomaly localization by modeling perceptual features |
| 10 | graphcore  | vig_ti_224_gelu | Pushing the limits of few-shot anomaly detection in industrial vision: GraphCore |
| 11 | igd | net_igd | Deep one-class classification via interpolated gaussian descriptor |
| 12 | padim  | resnet18, wide_resnet50 | Padim: a patch distribution modeling framework for anomaly detection and localization |
| 13 | patchcore  | resnet18, wide_resnet50 | Towards total recall in industrial anomaly detection |
| 14 | reverse | net_reverse | Anomaly detection via reverse distillation from one-class embedding |
| 15 | simplenet  | wide_resnet50 | SimpleNet: a simple network for image anomaly detection and localization |
| 16 | softpatch  | resnet18, wide_resnet50 | SoftPatch: unsupervised anomaly detection with noisy data |
| 17 | spade  | resnet18, wide_resnet50 | Sub-image anomaly detection with deep pyramid correspondences |
| 18 | stpm  | resnet18, wide_resnet50 | Student-teacher feature pyramid matching for anomaly detection |

## 3D Model
| No. | Method / -m | Net / -n | Paper Title |
| ------ | ------ | ------ | ------ |
| 1 | 3d_btf | -- | Back to the feature: classical 3D features are (almost) all you need for 3D anomaly detection | 
| 2 | 3d_ast | | AST: Asymmetric student-teacher networks for industrial anomaly detection |


## Run Example

> Vanilla / -v
```bash
python3 main.py -v -m cfa -n net_cfa -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m csflow -n net_csflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -v -m draem -n net_draem -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m fastflow -n net_fastflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m favae -n net_favae -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -v -m graphcore -n vig_ti_224_gelu -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -v -m igd -n net_igd -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m padim -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m patchcore -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -v -m reverse -n net_reverse -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m simplenet -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m spade -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m stpm -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
```

> Semi / -s
```bash
python3 main.py -s -m devnet -n net_devnet -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -s -m dra -n net_dra -d mvtec2d -tid 0 -vid 0 -g 1
```

> Fewshot / -f
```bash
python3 main.py -f -fe 1 -m patchcore -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -sp 0.1 -g 1 -fda -fnd 4 -fat rotation
python3 main.py -f -fe 1 -m _patchcore -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -sp 1 -fda -fnd 4 -g 1
python3 main.py -f -fe 1 -m csflow -n net_csflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m cfa -n net_cfa -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m draem -n net_draem -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m fastflow -n net_fastflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -f -fe 1 -m padim -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m favae -n net_favae -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -f -fe 1 -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m igd -n net_igd -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m reverse -n net_reverse -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m spade -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m stpm -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m graphcore -n vig_ti_224_gelu -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -f -fe 1 -m simplenet -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -g 1
```

> Continual / -c
```bash
python3 main.py -c -m patchcore -n wide_resnet50 -d mvtec2d -tid 0 1 -vid 0 1 -sp 0.001 -g 1
python3 main.py -c -m csflow -n net_csflow -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m cfa -n net_cfa -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m draem -n net_draem -d mvtec2d -tid 0 1 -vid 0 1 -g 2
python3 main.py -c -m fastflow -n net_fastflow -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 1 -vid 0 1  -g 1
python3 main.py -c -m padim -n resnet18 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m favae -n net_favae -d mvtec2d -tid 0 1 -vid 0 1  -g 1
python3 main.py -c -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m igd -n net_igd -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m reverse -n net_reverse -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m spade -n resnet18 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m stpm -n resnet18 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m dne -n vit_b_16 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m simplenet -n wide_resnet50 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
```

> Noisy / -z
```bash
python3 main.py -z -nr 0.1 -no -m softpatch -n wide_resnet50  -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -z -nr 0.1 -no -m patchcore -n wide_resnet50  -d mvtec2d  -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -z -nr 0.1 -no -m csflow -n net_csflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m cfa -n net_cfa -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m draem -n net_draem -d mvtec2d -tid 0 -vid 0 -g 2
python3 main.py -z -nr 0.1 -no -m fastflow -n net_fastflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -z -nr 0.1 -no -m padim -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m favae -n net_favae -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -z -nr 0.1 -no -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m reverse -n net_reverse -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m spade -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m stpm -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m igd -n net_igd -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m simplenet -n wide_resnet50  -d mvtec2d -tid 0 -vid 0 -g 1
```

> Transfer / -t
```bash
python3 main.py -t -ttn 8 -m reverse -n net_reverse -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m cfa -n net_cfa -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m csflow -n net_csflow -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m draem -n net_draem -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m fastflow -n net_fastflow -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m favae -n net_favae -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m padim -n resnet18 -d coad -tid 0 -vid 1 -g 1
python3 main.py -t -ttn 8 -m patchcore -n wide_resnet50 -d coad -tid 0 -vid 1 -g 1
python3 main.py -t -ttn 8 -m stpm -n resnet18 -d coad -tid 0 -vid 1 -g 1
python3 main.py -t -ttn 8 -m graphcore -n vig_ti_224_gelu -d coad -tid 0 -vid 1 -g 1
python3 main.py -t -ttn 8 -m simplenet -n wide_resnet50 -d coad -tid 0 -vid 1 -g 1
```


## Tutorial

### How to implement your own methods or datasets, i.e, integrating new methods into the open-iad project?

> Please refer to the following steps:

+ register your new method (e.g., MODEL, NET, DATASET, SETTING, SERVER) in [configuration/registration.py](configuration/registration.py)
+ add names of MODEL, NET, DATASET, SETTING into [configuration/config.py](configuration/config.py)
+ implement MODEL in [arch/_example.py](arch/_example.py) and [models/_example/net_example.py](models/_example/net_example.py)
+ put MODEL configuration in [configuration/1_model_base/_example.yaml](configuration/1_model_base/_example.yaml)
+ put DATASET configuration in [configuration/3_dataset_base/_example.yaml](configuration/3_dataset_base/_example.yaml)
+ implement DATASET in [dataset/_example.py](dataset/_example.py)
+ implement NEW SETTING in [data_io/data_holder.py](data_io/data_holder.py)
+ update output path of results in [tools/record_helper.py](tools/record_helper.py)
+ add NEW METHOD in [README.md](README.md)
+ shell command, "python3 main.py -v -m _example -n net_example -d _example -tid 0 -vid 0 -g 1"
