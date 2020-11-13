# A Unified Model for Real-Time Crop Recognition and Stem Localization Exploiting Cross-Task Feature Fusion

![IMAGE](https://github.com/ZhangXG001/Real-Time-Crop-Recognition-and-Stem-Localization/blob/master/IMG/network.jpg)

## Datesets

We provide our CWF-788 dataset with two different 
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
python trainseg.py
python trainstem.py
python trainfuse.py
```
You can get the well-trained model under the folder"model1".

### Test

If you want to test the model, you can run

```python
python testseg.py
python teststem.py
python testfuse.py
```

## Citation

If you think this work is helpful, please cite

Xiaoguang Z, Nan L, Luzhen G, et al. A Unified Model for Real-Time Crop Recognition and Stem Localization Exploiting Cross-Task Feature Fusion. 2020 IEEE International Conference on Realtime Computing and Robotics(RCAR).


## Acknowledgements
