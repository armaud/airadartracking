import torch
import torch.nn as nn
from datetime import datetime

from KalmanNet.Extended_sysmdl import SystemModel
import KalmanNet.config as config
from KalmanNet.utils import DataGen

from KalmanNet.EKF_test import EKFTest

from KalmanNet.Pipeline_EKF import Pipeline_EKF

from KalmanNet.parameters_radar import m1x_0, m2x_0, f, Q_structure, delta_t

m = 6 # State dimension: [px, vx, py, vy, pz, vz]
n = 3 # Observation dimension: [elevation, bearing, range]

measurement_type = "xyz" # Can be "xyz" for measurements in xyz or "ebr" measurements in elevation, bearing, range

if measurement_type == "ebr":
    from ssmodel_ebr import h_torch as h, h_inv_torch as h_inv, h_inv_two_meas_torch, get_radar_noise_covar
else:
    from ssmodel_xyz import h_torch as h, h_inv_torch as h_inv, h_inv_two_meas_torch, get_radar_noise_covar
    
print("Radar Performance Comparison Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 121
args.T_test = 121

### training parameters
args.n_steps = 2000
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

path_results = 'RTSNet/'
DatafolderName = 'dataset/'
if measurement_type == "ebr":
    DatafileName = 'torch_data.pt'
else:
    DatafileName = 'torch_data_xyz.pt'

# noise q and r
r2 = torch.tensor([1.0]) 
vdB = -20 # q2/r2 ratio in dB
v = 10**(vdB/10)
q2 = torch.mul(v, r2)

Q = q2[0] * Q_structure
R = get_radar_noise_covar()

#######################################
### System Model and Data Loading   ###
#######################################
sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)
sys_model.InitSequence(m1x_0, m2x_0)

[train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(DatafolderName + DatafileName)
if torch.isnan(train_input).any(): print("WARNING: train_input contains NaNs")
if torch.isnan(cv_input).any(): print("WARNING: cv_input contains NaNs")
print("Data Loaded successfully")

print("trainset size:", train_target.size())
print("testset size:", test_target.size())

#########################################
### Measurement-based Initialization ###
#########################################
print("Computing measurement-based initializations...")
def get_init(dataset_input):
    init_list = []
    for i in range(dataset_input.shape[0]):
        # Take first two measurements y_0, y_1
        y0, y1 = dataset_input[i, :, 0], dataset_input[i, :, 1]
        x0_est = h_inv_two_meas_torch(y0, y1, delta_t)
        if torch.isnan(x0_est).any():
             print(f"NaN in initialization at index {i}")
        init_list.append(x0_est)
    return torch.stack(init_list)

train_init = get_init(train_input)
cv_init = get_init(cv_input)
## For debugging only
print("x0_state", test_target[0, :, 0])
print("x0_meas", test_input[0, :, 0])
print("x1_state", test_target[0, :, 1])
print("x1_meas", test_input[0, :, 1])

test_init = get_init(test_input)


######################################
### Evaluate Filters and Smoothers ###
######################################

### Evaluate EKF
# print("Evaluate EKF")
# [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target, allStates=False, randomInit=True, test_init=test_init)


#########################
### Evaluate KalmanNet ###
#########################
print("Evaluate KalmanNet")
from KalmanNet.KalmanNet_nn import KalmanNetNN
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
KNet_Pipeline = Pipeline_EKF(strTime, "Models", "KalmanNet_Radar")
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setTrainingParams(args)
KNet_Pipeline.NNTrain(args.N_E, train_input, train_target, args.N_CV, cv_input, cv_target, randomInit=True, cv_init=cv_init, train_init=train_init, allStates=False)
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out] = KNet_Pipeline.NNTest(args.N_T, test_input, test_target, randomInit=True, test_init=test_init)

print("Comparison Finished")
