# AST: Asymmetric Student-Teacher Networks for Industrial Anomaly Detection

This is the code to the WACV 2023 paper "[Asymmetric Student-Teacher Networks for Industrial Anomaly Detection](https://arxiv.org/pdf/2210.07829.pdf)" by Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn and Bastian Wandt.

## Getting Started

You will need [Python 3.7.7](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) and installing the packages there.

Install packages with:

```
$ pip install -r requirements.txt
```

## Configure and Run

All configurations concerning data, model, training, visualization etc. can be made in _config.py_. The default configuration will run a training with paper-given parameters for MVTec 3D-AD.

The following steps guide you from your dataset to your evaluation:

* Set your _dataset_dir_ in _config.py_. This is the directory which contains the subdirectories of the classes you want to process. It should be configured whether the datasets contains 3D scans (set _use_3D_dataset_ in _config.py_). If 3D data is present (_use_3D_dataset=True_), the data structure from [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) is assumed. It can be chosen which data domain should be used for detection (set mode in _config.py_). If the dataset does not contain 3D data (_use_3D_dataset=False_), the data structure from the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) is assumed. Both dataset structures are described in the documentation of _load_img_datasets_ in _utils.py_.
* _preprocessing.py_: It is recommended to pre-extract the features beforehand to save training time. This script also preprocesses 3D scans. Alternatively, the raw images are used in training when setting _pre_extracted=False_ in _config.py_.
* _train_teacher.py_: Trains the teacher and saves the model to _models/..._
* _train_student.py_: Trains the student and saves the model to _models/..._
* _eval.py_: Evaluates the student-teacher-network (image-level results and localization/segmentation). Additionally, it creates ROC curves, anomaly score histograms and localization images.

Note: Due to a bug in the original implementation for the RGB+3D setting the performance on MVTec 3D-AD is about 2% better than originally reported.

## Credits

Some code of an old version of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.

## Citation
Please cite our paper in your publications if it helps your research. Even if it does not, you are welcome to cite us.

    @inproceedings { RudWeh2023,
    author = {Marco Rudolph and Tom Wehrbein and Bodo Rosenhahn and Bastian Wandt},
    title = {Asymmetric Student-Teacher Networks for Industrial Anomaly Detection},
    booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
    year = {2023},
    month = jan
    }

## License

This project is licensed under the MIT License.
