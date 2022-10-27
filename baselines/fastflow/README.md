# FastFlow

An unofficial pytorch implementation of [_FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows_](https://arxiv.org/abs/2111.07677) (Jiawei Yu et al.).

As the paper doesn't give all implementation details, it's kinda difficult to reproduce its result. A very close AUROC is achieved in this repo. But there are still some confusions and a lot of guesses:
- [ ] [Unmatched model A.d. parameter](https://github.com/gathierry/FastFlow/issues/2)
- [ ] [Unmentioned but critical LayerNorm](https://github.com/gathierry/FastFlow/issues/3)

_Really appreciate the inspiring [discussion](https://github.com/AlessioGalluccio/FastFlow/issues/14) with the community. Feel free to comment, raise new issues or PRs._

## Installation

```bash
pip install -r requirements.txt
```

## Data
We use [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) to verify the performance.

The dataset is organized in the following structure:
```
mvtec-ad
|- bottle
|  |- train
|  |- test
|  |- ground_truth
|- cable
|  |- train
|  |- test
|  |- ground_truth
...
```
## Train and eval
Take ResNet18 as example
```bash
# train
python main.py -cfg configs/resnet18.yaml --data path/to/mvtec-ad -cat [category]
# a folder named _fastflow_experiment_checkpoints will be created automatically to save checkpoints

# eval
python main.py -cfg configs/resnet18.yaml --data path/to/mvtec-ad -cat [category] --eval -ckpt _fastflow_experiment_checkpoints/exp[index]/[epoch#].pt
```

## Performance
As the training process is not stable, I paste both the performance of the last (500th) epoch and the best epoch.

| AUROC (last/best) | wide-resnet-50 | resnet18        | DeiT            | CaiT            |
| ----------------- | -------------- | --------------- | --------------- |-----------------|
| bottle            | 0.987/0.989    | 0.975/0.979     | 0.931/0.959     | 0.926/0.976     |
| cable             | 0.950/0.978    | 0.942/0.962     | 0.976/0.979     | 0.975/0.981     |
| capsule           | 0.987/0.989    | 0.979/0.985     | 0.982/0.988     | 0.987/0.990     |
| carpet            | 0.988/0.989    | 0.986/0.986     | 0.991/0.994     | 0.981/0.993     |
| grid              | 0.991/0.993    | 0.973/0.985     | 0.965/0.980     | 0.968/0.970     |
| hazel nut         | 0.957/0.984    | 0.922/0.963     | 0.982/0.990     | 0.981/0.992     |
| leather           | 0.995/0.996    | 0.991/0.996     | 0.991/0.994     | 0.994/0.996     |
| metal nut         | 0.968/0.986    | 0.950/0.966     | 0.980/0.988     | 0.977/0.984     |
| pill              | 0.968/0.977    | 0.955/0.968     | 0.977/0.989     | 0.984/0.990     |
| screw             | 0.969/0.987    | 0.952/0.957     | 0.990/0.990     | 0.991/0.993     |
| tile              | 0.955/0.971    | 0.916/0.951     | 0.966/0.966     | 0.946/0.972     |
| toothbrush        | 0.985/0.986    | 0.967/0.978     | 0.983/0.988     | 0.989/0.992     |
| transistor        | 0.956/0.975    | 0.970/0.975     | 0.959/0.970     | 0.967/0.969     |
| wood              | 0.948/0.964    | 0.894/0.954     | 0.960/0.963     | 0.950/0.959     |
| zipper            | 0.980/0.987    | 0.969/0.979     | 0.966/0.974     | 0.972/0.984     |
| __MEAN__          | __0.972/0.983__ | __0.956/0.972__ | __0.973/0.981__ | __0.973/0.983__ |
| Paper             | 0.981          | 0.972           | 0.981           | 0.985           |


