### Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection implementation (unofficial)
Unofficial pytorch implementation of  
Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection (STPM)  
\- Guodong Wang, Shumin Han, Errui Ding, Di Huang  (2021)  
https://arxiv.org/abs/2103.04257v2  

notice(21/06/02) :  
I rewrote a well-organized pytorch lightning version. Please check.  
https://github.com/hcw-00/STPM_pytorch_lightning


### Usage 
~~~
# python 3.6
pip install -r requirements.txt
python train.py --phase 'train or test' --dataset_path '...\mvtec_anomaly_detection\bottle' --project_path 'path\to\save\results'
~~~

### MVTecAD AUC-ROC score (mean of n trials)
| Category | Paper<br>(pixel-level) | This code<br>(pixel-level) | Paper<br>(image-level) | This code<br>(image-level) |
| :-----: | :-: | :-: | :-: | :-: |
| carpet | 0.988 | 0.987(1) | - | 0.985(1) |
| grid | 0.990 | 0.987(1) | - | 0.995(1) |
| leather | 0.993 | 0.989(1) | - | 1.000(1) |
| tile | 0.974 | 0.963(1) | - | 0.938(1) |
| wood | 0.972 | 0.949(1)| - | 0.993(1) |
| bottle | 0.988 | 0.982(1)| - | 1.000(1) |
| cable | 0.955 | 0.944(1) | - | 0.891(1) |
| capsule | 0.983 | 0.981(1) | - | 0.862(1) |
| hazelnut | 0.985 | 0.981(1) | - | 1.000(1) |
| metal nut | 0.976 | 0.968(1) | - | 0.999(1) |
| pill | 0.978 | 0.973(1) | - | 0.972(1) |
| screw | 0.983 | 0.985(1) | - | 0.900(1) |
| toothbrush | 0.989 | 0.984(1) | - | 0.875(1) |
| transistor | 0.825 | 0.800(1)| - | 0.916(1) |
| zipper | 0.985 | 0.978(1) | - | 0.879(1) |
| mean | 0.970 | 0.963(1) | 0.955 | 0.947(1) |


### Localization results   

![plot](./samples/bent_002_arr.png)
![plot](./samples/broken_003_arr.png)
![plot](./samples/metal_contamination_005_arr.png)

![plot](./samples/bent_lead_003_arr.png)
![plot](./samples/damaged_case_001_arr.png)

![plot](./samples/bent_wire_003_arr.png)
![plot](./samples/missing_cable_006_arr.png)

![plot](./samples/color_002_arr.png)
![plot](./samples/poke_008_arr.png)

![plot](./samples/combined_006_arr.png)
![plot](./samples/liquid_003_arr.png)
![plot](./samples/scratch_006_arr.png)

![plot](./samples/contamination_004_arr.png)
![plot](./samples/contamination_007_arr.png)

![plot](./samples/crack_005_arr.png)
![plot](./samples/cut_001_arr.png)
![plot](./samples/print_006_arr.png)

![plot](./samples/crack_010_arr.png)
![plot](./samples/faulty_imprint_006_arr.png)

![plot](./samples/hole_002_arr.png)
![plot](./samples/metal_contamination_008_arr.png)
![plot](./samples/thread_013_arr.png)