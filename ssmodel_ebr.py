import torch
import numpy as np

def h_torch(x):
    """
    Measurement function h(x) in PyTorch.
    x is [px, vx, py, vy, pz, vz]
    Returns [elevation, bearing, range] (Stone Soup CartesianToElevationBearingRange order)
    """
    px, py, pz = x[0], x[2], x[4]
    
    r = torch.sqrt(px**2 + py**2 + pz**2 + 1e-9)
    bearing = torch.atan2(py, px)
    elevation = torch.atan2(pz, torch.sqrt(px**2 + py**2 + 1e-9))
    
    return torch.stack([elevation, bearing, r])

def h_inv_torch(y):
    """
    Inverse measurement function h_inv(y) in PyTorch.
    y is [elevation, bearing, range]
    Returns [px, vx, py, vy, pz, vz]
    """
    el, az, r = y[0], y[1], y[2]
    
    px = r * torch.cos(el) * torch.cos(az)
    py = r * torch.cos(el) * torch.sin(az)
    pz = r * torch.sin(el)
    
    # Initialize velocity to zero
    return torch.tensor([[px], [0.0], [py], [0.0], [pz], [0.0]]).float()

def h_inv_two_meas_torch(y0, y1, delta_t):
    """
    Inverse measurement function h_inv(y) using two measurements to estimate velocity.
    y0, y1 are [elevation, bearing, range]
    Returns [px, vx, py, vy, pz, vz]
    """
    el0, az0, r0 = y0[0], y0[1], y0[2]
    el1, az1, r1 = y1[0], y1[1], y1[2]
    
    # Position 0
    px0 = r0 * torch.cos(el0) * torch.cos(az0)
    py0 = r0 * torch.cos(el0) * torch.sin(az0)
    pz0 = r0 * torch.sin(el0)
    
    # Position 1
    px1 = r1 * torch.cos(el1) * torch.cos(az1)
    py1 = r1 * torch.cos(el1) * torch.sin(az1)
    pz1 = r1 * torch.sin(el1)
    
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
    elevation_std = np.deg2rad(3)
    bearing_std = np.deg2rad(0.15)
    range_std = 25.0
    
    R = torch.diag(torch.tensor([elevation_std**2, bearing_std**2, range_std**2])).float()
    return R