# Multi-Target tracking simulation

## Generating dataset
Create dataset in multiple pickle files. This format is useful for testing classical tracking algorithms.
```python
python create_dataset.py
```

## Convert dataset to tensors
For AI based tracking (In-progress), data in tensor is preferred. There are two converters. The *data_converter.py* converter is useful when we want measurements in range, azimuth, elevation format.
```python
python data_converter.py
```
The *data_converter_xyz.py* is useful when we want measurements in x,y,z coordinates. 
```python
python data_converter_xyz.py
```
We are using both formats to see which one is more suitable.
## Running tracker on a single dataset file

```python
python tracking_simulator.py --dataset path/to/data-pickle-file
```
## KalmanNet Tracker (In-progress)
```python
python tracker_kalmannet.py
```

