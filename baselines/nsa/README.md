# Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization

**Abstract:** We introduce a simple and intuitive self-supervision task, Natural Synthetic
Anomalies (NSA), for training an end-to-end model for anomaly detection and
localization using only normal training data. NSA integrates Poisson image
editing to seamlessly blend scaled patches of various sizes from separate
images. This creates a wide range of synthetic anomalies which are more similar
to natural sub-image irregularities than previous data-augmentation strategies
for self-supervised anomaly detection. We evaluate the proposed method using
natural and medical images. Our experiments with the MVTec AD dataset show that
a model trained to localize NSA anomalies generalizes well to detecting
real-world a priori unknown types of manufacturing defects. Our method achieves
an overall detection AUROC of 97.2 outperforming all previous methods that
learn without the use of additional datasets.

Please see our arXiv preprint for more details: https://arxiv.org/abs/2109.15222.

## Data
The NIH chest X-ray data can be downloaded [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/371647823217).

The MVTec AD dataset can be downloaded [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).

## Training
The method for generating self-supervised examples is defined in `self_sup_data/self_sup_tasks.py`. This is applied dynamically to the images while loading similar to other data-augmentation transforms (see the corresponding `Dataset` classes).
To train anomaly detection models for chest X-ray or MVTec AD data use `train_mvtec.py` and `train_chest_xray.py`. E.g.
```
python3 train_chest_xray.py -s Shift-Intensity-M -d '/path/to/cxr/images/' -o '/where/to/save/models/' -l 'self_sup_data/chest_xray_lists/norm_MaleAdultPA_train_curated_list.txt'
python3 train_mvtec.py -s Shift-Intensity-923874273 -d '/path/to/mvtec_ad/images/' -o '/where/to/save/models/' -n zipper
```

## Evaluation
The evaluation procedures are defined in `experiments/mvtec_tasks.py` and `experiments/chest_xray_tasks.py`. The evaluation notebooks (`mvtec_evaluation.ipynb` and `chestxray_evaluation.ipynb`) can be used to run the evaluation and generate tables for sample-level and pixel-level AUROC and AU-PRO where applicable.
