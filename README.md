# IM-VAD: Vision Anomaly Detection Benchmark in Industrial Manufacturing
## Preliminary  

```bash
pytorch 1.10.1 or 1.8.1
conda activate ad-3d
pip3 install -r requirements.txt

# example
pip install scikit-image
pip instll scikit-learn
pip install opencv-python
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
├── data_io # dataset processing and loading data interface
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

## Dataset (--dataset / -d)

> 2D: mvtec2d, mpdd, mvtecloco, mtd, btad, mvtec2df3d, imad_hardware_parts

> 3D: mvtec3d

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

## 2D Model
| Method / -m | Net / -n | Paper Title|
| ------ | ------ | ------ |
| cfa | net_cfa | CFA: Coupled-hypersphere-based feature adaptation for target-oriented anomaly localization |
| csflow | net_csflow | Fully convolutional cross-scale-flows for image-based defect detection |
| cutpaste | vit_b_16 | Cutpaste: Self-supervised learning for anomaly detection and localization |
| devnet | net_devnet | Explainable deep few-shot anomaly detection with deviation networks |
| dne | vit_b_16 | Towards continual adaptation in industrial anomaly detection |
| dra | net_dra | Catching both gray and black swans: open-set supervised anomaly detection |
| draem | net_draem | Draem: A discriminatively trained reconstruction embedding for surface anomaly detection |
| fastflow | net_fastflow | Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows |
| favae | net_favae | Anomaly localization by modeling perceptual features |
| igd | net_igd | Deep one-class classification via interpolated gaussian descriptor |
| padim  | resnet18, wide_resnet50 | Padim: a patch distribution modeling framework for anomaly detection and localization |
| patchcore  | resnet18, wide_resnet50 | Towards total recall in industrial anomaly detection |
| reverse (rd4ad) | net_reverse | Anomaly detection via reverse distillation from one-class embedding |
| spade  | resnet18, wide_resnet50 | Sub-image anomaly detection with deep pyramid correspondences |
| stpm  | resnet18, wide_resnet50 | Student-teacher feature pyramid matching for anomaly detection |
| graphcore  | vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu | Pushing the limits of few-shot anomaly detection in industrial vision: GraphCore |

## 3D Model
| Method / -m | Net / -n | Paper Title |
| ------ | ------ | ------ |
| 3d_btf | -- | Back to the feature: classical 3D features are (almost) all you need for 3D anomaly detection | 
| 3d_ast | | AST: Asymmetric student-teacher networks for industrial anomaly detection |


## Run Example

> Vanilla / -v
```bash
python3 main.py -p c2d -v -m patchcore -n resnet18 -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -p c2d -v -m csflow -n net_csflow -d mvtec2d -tid 11 -vid 11 -g 1
python3 main.py -p c2d -v -m cfa -n net_cfa -d mvtec2d -tid 11 -vid 11 -g 7
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
python3 main.py -p c2d -v -m graphcore -n vig_ti_224_gelu -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
```

> Continual / -c
```bash
python3 main.py -p c2d -c -m patchcore -n resent18 -d mvtec2d -tid 0 1 -vid 0 1 -sp 0.001 -g 1
python3 main.py -p c2d -c -m csflow -n net_csflow -d mvtec2d -tid 10 11 -vid 10 11 -g 1
python3 main.py -p c2d -c -m cfa -n net_cfa -d mvtec2d -tid 10 11 -vid 10 11 -g 7
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

> Fewshot / -f
```bash
python3 main.py -p c2d -f -fda --fewshot-exm 4 -m patchcore -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -sp 0.1 -g 0 -fnd 4 -fat rotation
python3 main.py -p c2d -f --fewshot-exm 1 --fewshot-num-dg 4 -m _patchcore -n resnet18 -d mvtec2d -tid 0 -vid 0 -sp 1 -g 1
python3 main.py -p c2d -f --fewshot-exm 1 -m patchcore -n resnet18 -d mvtec2d -tid 0 -vid 0 -sp 1 -g 1
python3 main.py -p c2d -f --fewshot-exm 1 -m csflow -n net_csflow -d mvtec2d -tid 11 -vid 11 -g 1
python3 main.py -p c2d -f --fewshot-exm 1 -m cfa -n net_cfa -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m draem -n net_draem -d mvtec2d -tid 11 -vid 11 -g 2
python3 main.py -p c2d -f --fewshot-exm 1 -m fastflow -n net_fastflow -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m cutpaste -n vit_b_16  -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m padim -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m favae -n net_favae -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m cutpaste -n vit_b_16 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m igd -n net_igd -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m reverse -n net_reverse -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m spade -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -f --fewshot-exm 1 -m stpm -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
```

> Semi / -s
```bash
python3 main.py -p c2d -s -m devnet -n net_devnet -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -p c2d -s -m dra -n net_dra -d mvtecloco -tid 0 -vid 0 -g 1
```

> Noisy / -z
```bash
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m patchcore -n resnet18 -d mvtec2d -tid 0 -vid 1 -sp 0.001 -g 1
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m csflow -n net_csflow -d mvtec2d -tid 11 -vid 11 -g 1
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m cfa -n net_cfa -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m draem -n net_draem -d mvtec2d -tid 11 -vid 11 -g 2
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m fastflow -n net_fastflow -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m cutpaste -n vit_b_16  -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m padim -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m favae -n net_favae -d mvtec2d -tid 11 -vid 11  -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m cutpaste -n vit_b_16 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m reverse -n net_reverse -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m spade -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m stpm -n resnet18 -d mvtec2d -tid 11 -vid 11 -g 7
python3 main.py -p c2d -z --noisy-ratio 0.1 --noisy-overlap -m igd -n net_igd -d mvtec2d -tid 11 -vid 11 -g 7
```
