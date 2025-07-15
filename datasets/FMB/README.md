# FMB DATASET

The dataset of FMB can be downloaded from official repository [FMB](https://github.com/JinyuanLiu-CV/SegMiF)

Unzip the train e test zip files in a folder called FMB
```
unzip datasets/test.zip -d datasets/FMB
unzip datasets/train.zip -d datasets/FMB
```
Within FMB folder execute the following command
```
python extract_FMB_val.py
python extract_val_label_infrared_color.py
```