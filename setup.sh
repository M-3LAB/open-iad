#!/bin/bash
pip install -r requirements.txt

cd third_party/chamfer3D/
python3 setup.py install

cd ../emd/
python3 setup.py install

