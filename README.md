# A Unified Model for Real-Time Crop Recognition and Stem Localization Exploiting Cross-Task Feature Fusion

![IMAGE](https://github.com/ZhangXG001/Real-Time-Crop-Recognition-and-Stem-Localization/blob/master/IMG/f68d89a1515a72e32d3a9115b92b893.jpg)

## Datesets

We provide our CWF-788 dataset with two different labels(segmentation and stem-localization),baiduyun link: https://pan.baidu.com/s/1zJ1ssf93edHOGqilRaO2ig
password：1111 
## Usage

Please install Tensorflow and required packages first.

### Download our code.

```python
git clone https://github.com/ZhangXG001/Real-Time-Crop-Recognition-and-Stem-Localization.git
```

### Create .csv files of dataset.

You can run ``` .../python csv_generator.py ```to create .csv files of dataset.

### Train

If you want to train the model,you can run

```python
cd .../Real-Time-Crop-Recognition-and-Stem-Localization-master
python trainseg.py(only for segmentation)
python trainstem.py(only for stem-localization)
python trainfuse.py(segmentation and stem-localization at the same time)
```
You can get the well-trained model under the folder"model1".

### Test

If you want to test the model, you can run

```python
python testseg.py(tset segmentation model)
python teststem.py(tset stem-localization model)
python testfuse.py(tset segmentation and stem-localization model)
```

## Citation

If you think this work is helpful, please cite

Xiaoguang Z, Nan L, Luzhen G, et al. A Unified Model for Real-Time Crop Recognition and Stem Localization Exploiting Cross-Task Feature Fusion. 2020 IEEE International Conference on Realtime Computing and Robotics(RCAR).


## Acknowledgements
