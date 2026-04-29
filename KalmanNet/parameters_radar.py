import torch
import math
from torch import autograd

#########################
### Design Parameters ###
#########################
m = 6 # State dimension: [px, vx, py, vy, pz, vz]
n = 4 # Observation dimension: [range, azimuth, elevation, doppler]

m1x_0 = torch.tensor([[100.0], [1.0], [100.0], [1.0], [100.0], [1.0]]) 
m2x_0 = 10 * torch.eye(m)

### Time step
delta_t = 1.0

######################################################
### State evolution function f (Constant Velocity) ###
######################################################
def f(x):
    # x is [px, vx, py, vy, pz, vz]
    F = torch.tensor([
        [1, delta_t, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, delta_t, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, delta_t],
        [0, 0, 0, 0, 0, 1]
    ]).float()
    
    if x.dim() == 1:
        return torch.matmul(F, x)
    return torch.matmul(F, x)

##################################################
### Observation function h (Range-Doppler Radar) ###
##################################################
def h(x):
    # x is [px, vx, py, vy, pz, vz]
    px, vx, py, vy, pz, vz = x[0], x[1], x[2], x[3], x[4], x[5]
    
    range_val = torch.sqrt(px**2 + py**2 + pz**2 + 1e-9)
    azimuth = torch.atan2(py, px)
    elevation = torch.atan2(pz, torch.sqrt(px**2 + py**2 + 1e-9))
    doppler = (px*vx + py*vy + pz*vz) / range_val
    
    y = torch.stack([range_val, azimuth, elevation, doppler])
    return y

###############################################
### process noise Q and observation noise R ###
###############################################
Q_structure = torch.eye(m)
R_structure = torch.eye(n)

##################################
### Utils for non-linear cases ###
##################################
def h_inv(y):
    # y is [range, azimuth, elevation, doppler]
    r, az, el, d = y[0], y[1], y[2], y[3]
    
    px = r * torch.cos(el) * torch.cos(az)
    py = r * torch.cos(el) * torch.sin(az)
    pz = r * torch.sin(el)
    
    # Doppler is radial velocity: vr = (px*vx + py*vy + pz*vz)/r
    # We can at least initialize velocity components based on radial velocity if we assume direction
    # For now, let's just initialize position and set velocity to a small default or zero
    vx = d * torch.cos(el) * torch.cos(az)
    vy = d * torch.cos(el) * torch.sin(az)
    vz = d * torch.sin(el)
    
    x = torch.tensor([px, vx, py, vy, pz, vz]).float()
    return x.view(-1, 1)

def getJacobian(x, g):
    # Ensure x is 1D for jacobian tool
    if x.dim() > 1:
        y = x.view(-1)
    else:
        y = x
        
    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1, m)
    return Jac
