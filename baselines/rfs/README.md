[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anomaly-detection-of-defect-using-energy-of/few-shot-anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/few-shot-anomaly-detection-on-mvtec-ad?p=anomaly-detection-of-defect-using-energy-of)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anomaly-detection-of-defect-using-energy-of/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=anomaly-detection-of-defect-using-energy-of)

# RFS-Energy-Anomaly-Detection-of-Defect
Implemention of the paper ["Anomaly Detection of Defect using Energy of Point Pattern Features within Random Finite Set Framework"](https://arxiv.org/abs/2108.12159)

https://arxiv.org/abs/2108.12159

<p align="center">
<img src="https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/blob/main/Img/proposed_approach.png" width="1024">
</p>

## Anomaly Detection of Defect using Energy of Point Pattern Features within Random Finite Set Framework Results and Comparisons

- AUC Performance on MVTec-AD
94.1 % AUC of the ROC curve
-Comparison provided against [DifferNet](https://arxiv.org/pdf/2008.12577.pdf), [1-NN](https://arxiv.org/pdf/1811.08495.pdf), [OCSVM](https://discovery.ucl.ac.uk/id/eprint/10062495/), [DSEBM](https://arxiv.org/pdf/1605.07717.pdf), [GANomaly](https://arxiv.org/pdf/1805.06725.pdf), and [GeoTrans](https://arxiv.org/pdf/1805.10917.pdf).

<p align="center">
<img src="https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/blob/main/Img/AUC_MVTec.png" width="650">
</p>

 ROC Performance MVTec AD
<p align="center">
<img src="https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/blob/main/Img/roc_2.svg" width="650">
</p>
<p align="center">
<img src="https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/blob/main/Img/roc_3.svg" width="650">
</p>

 #Few-shot Experimental Results for MVTec AD 
  - AUC=89.02 % for Ten Shot
  - Comparison provided against [DifferNet](https://arxiv.org/pdf/2008.12577.pdf), [DROCC](https://arxiv.org/pdf/2002.12718.pdf), [PatchSVDD](https://arxiv.org/pdf/2011.08785.pdf),[DeepSVDD](http://proceedings.mlr.press/v80/ruff18a.html),[GeoTrans](https://arxiv.org/pdf/1805.10917.pdf), and [HTDGM](https://arxiv.org/pdf/2104.14535.pdf).
  
  <p align="center">
<img src="https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/blob/main/Img/MVTec_fewshot_results.png" width="650">
</p>
MVTecAD AUC vs shots
<p align="center">
<img src="https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/blob/main/Img/MVTec_AUC_overshots.png" width="650">
</p>

Samples of D2-Net Features 
<p align="center">
<img src="https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/blob/main/Img/samples_d2net_features_Page_2.jpg" width="650">
</p>

## The source code of the paper is placed under [code](https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect/tree/main/code)
