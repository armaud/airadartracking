import torch
import numpy as np

def h_torch(x):
    """
    Measurement function h(x) in PyTorch.
    x is [px, vx, py, vy, pz, vz]
    Returns [x, y, z] 
    """
    px, py, pz = x[0], x[2], x[4]
    
    return torch.stack([px, py, pz])

def h_inv_torch(y):
    """
    Inverse measurement function h_inv(y) in PyTorch.
    y is [px, py, pz]
    Returns [px, vx, py, vy, pz, vz]
    """
    
    # Initialize velocity to zero
    return torch.tensor([[y[0]], [0.0], [y[1]], [0.0], [y[2]], [0.0]]).float()

def h_inv_two_meas_torch(y0, y1, delta_t):
    """
    Inverse measurement function h_inv(y) using two measurements to estimate velocity.
    y0, y1 are [elevation, bearing, range]
    Returns [px, vx, py, vy, pz, vz]
    """
    px0, py0, pz0 = y0[0], y0[1], y0[2]
    px1, py1, pz1 = y1[0], y1[1], y1[2]
    
      
    # Velocity estimation
    vx = (px1 - px0) / delta_t
    vy = (py1 - py0) / delta_t
    vz = (pz1 - pz0) / delta_t
    
    # Back-propagate position to t=0 (given p0 was at t=1)
    px_init = px0 - vx * delta_t
    py_init = py0 - vy * delta_t
    pz_init = pz0 - vz * delta_t
    
    return torch.tensor([[px_init], [vx], [py_init], [vy], [pz_init], [vz]]).float()

def get_radar_noise_covar():
    """
    Returns the measurement noise covariance R as a PyTorch tensor.
    Matches the noise values in RadarEmulator.
    """
    x_std = 25.0
    y_std = 25.0
    z_std = 25.0
    
    R = torch.diag(torch.tensor([x_std**2, y_std**2, z_std**2])).float()
    return R
