import numpy as np
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.base import Property
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.dataassociator.neighbour import NearestNeighbour
from radar_sim import RadarMeasurementModel
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.array import CovarianceMatrix
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange

# class Initiator(SimpleMeasurementInitiator):
#     def initiate(self, detections, timestamp, **kwargs):
#         MAX_DEV = 500.
#         tracks = set()
#         measurement_model = self.measurement_model
#         for detection in detections:
#             state_vector = measurement_model.inverse_function(
#                             detection)
#             model_covar = measurement_model.covar()

#             el_az_range = np.sqrt(np.diag(model_covar))  # elev, az, range

#             std_pos = detection.state_vector[2, 0]*el_az_range[1]
#             stdx = np.abs(std_pos*np.sin(el_az_range[1]))
#             stdy = np.abs(std_pos*np.cos(el_az_range[1]))
#             stdz = np.abs(detection.state_vector[2, 0]*el_az_range[0])
#             if stdx > MAX_DEV:
#                 print('Warning - X Deviation exceeds limit!!')
#             if stdy > MAX_DEV:
#                 print('Warning - Y Deviation exceeds limit!!')
#             if stdz > MAX_DEV:
#                 print('Warning - Z Deviation exceeds limit!!')
#             C0 = np.diag(np.array([stdx, 50.0, stdy, 50.0, stdz, 10.0])**2)

#             tracks.add(Track([GaussianStateUpdate(
#                 state_vector,
#                 C0,
#                 SingleHypothesis(None, detection),
#                 timestamp=detection.timestamp)
#             ]))
#         return tracks


class TrackerManager:
    def __init__(self):
        self.tracks = set()
        
        # Transition
        self.transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(1.0), ConstantVelocity(1.0), ConstantVelocity(0.1)]
        )
        # 1.0, 1.0, 0.1
        # Measurement
        # meas_cov = np.diag([200**2, np.radians(3.0)**2, np.radians(3.0)**2, 100.0**2])
        # self.measurement_model = RadarMeasurementModel(center_freq=10e9, prf=5000)
        # self.measurement_model.noise_covar = meas_cov
        
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
        
        # Components
        # Create an Unscented Kalman Predictor
        self.predictor = UnscentedKalmanPredictor(self.transition_model)

        # Create an Unscented Kalman Updater, note our sensor adds a measurement model to detections
        self.updater = UnscentedKalmanUpdater(measurement_model=None)
        
        self.hypothesiser = DistanceHypothesiser(self.predictor, self.updater, measure=Mahalanobis(), missed_distance=2)

        self.associator = GNNWith2DAssignment(self.hypothesiser)
        
        # self.hypothesiser = PDAHypothesiser(
        #     self.predictor, 
        #     self.updater, 
        #     prob_detect=0.9, 
        #     clutter_spatial_density=1e-9,
        #     prob_gate=0.999
        # )
        # self.associator = JPDA(self.hypothesiser)

        self.deleter = UpdateTimeStepsDeleter(time_steps_since_update=5)
        
        min_detections = 3  # number of detections required to begin a track
        # initiator_prior_state = GaussianState(
        #     state_vector=np.array([[0], [0], [0], [0], [0], [0]]),
        #     covar=np.diag([300, 500, 300, 500, 300, 500])**2
        # )
        initiator_prior_state = GaussianState(
            state_vector=np.array([[0], [0], [0], [0], [0], [0]]),
            covar=np.diag([0, 500, 0, 500, 0, 500])**2
        )
        # covar=np.diag([1e5, 500, 1e5, 500, 1e5, 500])**2
        # initiator_meas_model = CartesianToElevationBearingRange(
        #     ndim_state=6,
        #     mapping=np.array([0, 2, 4]),
        #     noise_covar=noise_covar
        # )
        
        # self.initiator = MultiMeasurementInitiator(
        #     prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
        #     measurement_model=measurement_model,
        #     deleter=deleter,
        #     data_associator=data_associator,
        #     updater=updater,
        #     min_points=2,
        #     )

        self.initiator = MultiMeasurementInitiator(
            prior_state=initiator_prior_state,
            measurement_model=self.measurement_model,
            deleter=self.deleter,
            data_associator=NearestNeighbour(self.hypothesiser),
            updater=self.updater,
            min_points=min_detections,
            updates_only=True
        )
        

    def update(self, detections, current_time):
        
        all_tracks = set()

        # for n, measurements in enumerate(detections):
        # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
        hypotheses = self.associator.associate(self.tracks,
                                            detections,
                                            current_time)
        associated_measurements = set()
        for track in self.tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = self.updater.update(hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:  # When data associator says no detections are good enough, we'll keep the prediction
                track.append(hypothesis.prediction)

        # Carry out deletion and initiation
        self.tracks -= self.deleter.delete_tracks(self.tracks)
        self.tracks |= self.initiator.initiate(set(detections) - associated_measurements, current_time)
        all_tracks |= self.tracks
        
        return self.tracks
        