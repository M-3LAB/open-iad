# Anomaly-Detection-RGBD (ad-3d)
##Preliminary  

pytorch 1.10.1 or 1.8.1\
scikit-image, pip install scikit-image\
scikit-learn, pip instll scikit-learn\
opencv, pip install opencv-python

## Preliminary
> Dependency

```bash
conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
```bash
pip3 install -r requirements.txt
```
## Centralized Training
```bash
python3 centralized_training.py --dataset 'mvtec2d' --model 'patchcore2d' --data-path '/disk1/mvtec/2D' --continual --num-task 5
python3 centralized_training.py --dataset 'mvtec3d' --model 'patchcore3d' --data-path '/disk1/mvtec/3D' --continual --num-task 5
```
## Genate File with Format .xyzrgb and Visualization
```bash
python3 legacy_code/MVTec3D-ADPC.py 
python3 legacy_code/MVTec3D-PCVis.py 
```