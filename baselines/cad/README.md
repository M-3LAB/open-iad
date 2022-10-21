# Continual Anomaly Detection

## Datasets
To train on the MVtec Anomaly Detection dataset [download](https://www.mvtec.com/company/research/datasets/mvtec-ad) 
the data and extract it. For the additional Magnetic Tile Defects dataset, we [download](https://github.com/abin24/Magnetic-tile-defect-datasets.) the data then run **make_mtd_ano.py** for anomaly detection.

## Enviroment setup
```
pip install -r requirements.txt
```

## Getting pretrained ViT model
ViT-B/16 model used in this paper can be downloaded at [here](https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz).

## Run
We provide the configuration file to run CAD on multiple benchmarks in `configs`.

```
python main.py --config-file ./configs/adc.yaml  --data_dir ../datasets/mvtec --mtd_dir ../datasets/mtd_ano_mask
```



