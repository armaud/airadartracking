# Some general imports and set up
from datetime import datetime
from datetime import timedelta

import numpy as np
import torch

# Stone Soup imports:
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from matplotlib import pyplot as plt

from stonesoup.platform.base import FixedPlatform
from stonesoup.sensor.radar.radar import AESARadar, RadarElevationBearingRange
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.sensor.radar.beam_shape import Beam2DGaussian
from stonesoup.sensor.radar.beam_pattern import BeamSweep


class RadarEmulator:
    """
    Simulates a Pulse Doppler Radar with Range-Doppler map generation and CFAR detection.
    """
    def __init__(self, start_time, center_freq=10e9, bandwidth=5e6, prf=5000, 
                 n_pulses=64, noise_power=1.0, probability_false_alarm=1e-4,
                 transmit_power=10000.0, antenna_gain_db=30.0):
        self.fc = center_freq
        self.c = 3e8
        self.bw = bandwidth
        self.prf = prf
        self.n = n_pulses
        self.noise_power = noise_power # Expected to be unitless or relative noise floor
        self.pfa = probability_false_alarm
        self.pt = transmit_power
        self.gt_db = antenna_gain_db
        self.gt = 10**(self.gt_db/10)
        
        # Resolution
        self.delta_r = self.c / (2 * self.bw)
        self.delta_v = (self.c * self.prf) / (2 * self.fc * self.n)
        self.max_range = self.c / (2 * self.prf) # Unambiguous range (simple) 
        # Actually standard equation is c / (2 * PRF) but let's stick to a window
        self.max_range = 50000 # fixed window for sim
        
        self.n_range_bins = int(self.max_range / self.delta_r)
        self.n_doppler_bins = self.n
 
        # Create the initial state (position, time), notice it is set to the simulation start time defined earlier
        self.platform_state = State(StateVector([[0], [0], [0]]), start_time)

        # create our fixed platform
        self.platform = FixedPlatform(states=self.platform_state,
                         position_mapping=(0, 1, 2))

        self.noise_covar = CovarianceMatrix(np.array(np.diag([np.deg2rad(3)**2,
                                                 np.deg2rad(0.15)**2,
                                                 25**2])))
        self.noise_covar = CovarianceMatrix(np.array(np.diag([np.deg2rad(3)**2,
                                                 np.deg2rad(0.15)**2,
                                                 25**2])))
        # this radar measures range with an accuracy of +/- 25m, and elevation accuracy +/- 3
        # degrees and bearing accuracy of +/- 0.15 degrees

        self.measurement_model = CartesianToElevationBearingRange(
            ndim_state=6,
            mapping=(0, 2, 4),  # x, y, z
            noise_covar=self.noise_covar
        )
        
        # self.radar = AESARadar(
        #     measurement_model=self.measurement_model,
        #     beam_shape=Beam2DGaussian,
        #     beam_transition_model=BeamSweep,
        #     beam_width=np.radians(3),
        #     duty_cycle=0.1,
        #     band_width=bandwidth,
        #     frequency=center_freq,
        #     antenna_gain=antenna_gain_db,
        #     receiver_noise=1)
        self.radar = RadarElevationBearingRange(ndim_state=6,
                                   position_mapping=(0, 2, 4),
                                   noise_covar=self.noise_covar)
        self.platform.add_sensor(self.radar)


    
    def get_detections(self, targets, current_time):
        gtruths = set(targets)
        mset = []
        # for target in targets:
        #     measurement = self.radar.measure(ground_truths=target,noise=False)
        #     mset.append(measurement)
        measurement = self.radar.measure(ground_truths=gtruths,noise=False)
        return measurement

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