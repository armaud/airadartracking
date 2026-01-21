import numpy as np
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
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

# class RadarMeasurementModel(MeasurementModel):
#     """
#     Non-linear measurement model for [Range, Azimuth, Elevation, RangeRate]
#     State: [x, vx, y, vy, z, vz]
#     """
#     center_freq: float = Property(default=10e9, doc="Radar center frequency")
#     prf: float = Property(default=5000, doc="Pulse Repetition Frequency")
    
#     #    out = np.random.multivariate_normal(np.zeros(self.ndim_meas), self.noise_covar, num_samples).T
#     # But wait, we need to fix the class definition first.
    
#     def __init__(self, ndim_state=6, mapping=(0, 2, 4), center_freq=10e9, prf=5000):
#         super().__init__(ndim_state=ndim_state, mapping=mapping)
#         self.noise_covar = np.diag([10**2, np.radians(1)**2, np.radians(1)**2, 5**2]) # Widen velocity gate
#         self.c = 3e8
#         self.lambda_w = self.c / self.center_freq
#         self.v_ambig = self.prf * self.lambda_w / 2.0

#     def covar(self, **kwargs):
#         return self.noise_covar
        
#     def function(self, state, noise=False, **kwargs):
#         x = state.state_vector[0, 0]
#         y = state.state_vector[2, 0]
#         z = state.state_vector[4, 0]
#         vx = state.state_vector[1, 0]
#         vy = state.state_vector[3, 0]
#         vz = state.state_vector[5, 0]

#         r = np.sqrt(x**2 + y**2 + z**2)
#         az = np.arctan2(y, x)
#         el = np.arcsin(z / r) if r != 0 else 0
        
#         # Range Rate
#         if r == 0:
#             rr = 0
#         else:
#             rr = (x*vx + y*vy + z*vz) / r
            
#         # Aliasing (Wrapping)
#         # We need to wrap the predicted range rate into the interval [-v_ambig/2, v_ambig/2]
#         # This assumes the measurement is always aliased this way
        
#         rr_wrapped = rr - self.v_ambig * round(rr / self.v_ambig)

#         out = np.array([r, az, el, rr_wrapped]).reshape(4, 1)
        
#         if noise:
#             out += self.rvs()
            
#         return out

#     def jacobian(self, state, **kwargs):
#         # Note: Jacobian technically ignores wrapping discontinuities, which is standard for EKF
#         # unless we are right on the edge.
#         x = state.state_vector[0, 0]
#         y = state.state_vector[2, 0]
#         z = state.state_vector[4, 0]
#         vx = state.state_vector[1, 0]
#         vy = state.state_vector[3, 0]
#         vz = state.state_vector[5, 0]
        
#         r2 = x**2 + y**2 + z**2
#         r = np.sqrt(r2)
#         r3 = r2 * r
        
#         if r == 0: return np.zeros((4, 6))

#         xy2 = x**2 + y**2
        
#         # Row 0: R
#         dr_dx = x/r
#         dr_dy = y/r
#         dr_dz = z/r
        
#         # Row 1: Az
#         if xy2 == 0:
#             daz_dx, daz_dy = 0, 0
#         else:
#             daz_dx = -y / xy2
#             daz_dy = x / xy2
        
#         # Row 2: El
#         pre = r / np.sqrt(xy2) if xy2 > 0 else 0
#         del_dx = pre * (-z*x/r3)
#         del_dy = pre * (-z*y/r3)
#         del_dz = pre * ((x**2+y**2)/r3)
        
#         # Row 3: RR
#         drr_dx = (vx*r - ((x*vx + y*vy + z*vz)*x/r))/r2
#         drr_dy = (vy*r - ((x*vx + y*vy + z*vz)*y/r))/r2
#         drr_dz = (vz*r - ((x*vx + y*vy + z*vz)*z/r))/r2
        
#         drr_dvx = x/r
#         drr_dvy = y/r
#         drr_dvz = z/r
        
#         H = np.zeros((4, 6))
#         H[0, 0] = dr_dx; H[0, 2] = dr_dy; H[0, 4] = dr_dz
#         H[1, 0] = daz_dx; H[1, 2] = daz_dy
#         H[2, 0] = del_dx; H[2, 2] = del_dy; H[2, 4] = del_dz
#         H[3, 0] = drr_dx; H[3, 1] = drr_dvx; H[3, 2] = drr_dy; H[3, 3] = drr_dvy; H[3, 4] = drr_dz; H[3, 5] = drr_dvz
        
#         return H

#     def rvs(self, num_samples=1, **kwargs):
#         out = np.random.multivariate_normal(np.zeros(self.ndim_meas), self.noise_covar, num_samples).T
#         if num_samples == 1:
#             return out[:, 0]
#         return out

#     def pdf(self, state1, state2, **kwargs):
#         return super().pdf(state1, state2, **kwargs)
    
#     @property
#     def ndim_meas(self):
#         return 4

class TrackerManager:
    def __init__(self):
        self.tracks = set()
        
        # Transition
        self.transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(1.0), ConstantVelocity(1.0), ConstantVelocity(0.1)]
        )
        
        # Measurement
        meas_cov = np.diag([200**2, np.radians(3.0)**2, np.radians(3.0)**2, 100.0**2])
        self.measurement_model = RadarMeasurementModel(center_freq=10e9, prf=5000)
        self.measurement_model.noise_covar = meas_cov
        
        # Components
        self.predictor = ExtendedKalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(self.measurement_model)
        
        self.hypothesiser = PDAHypothesiser(
            self.predictor, 
            self.updater, 
            prob_detect=0.9, 
            clutter_spatial_density=1e-9,
            prob_gate=0.999
        )
        self.associator = JPDA(self.hypothesiser)

        self.deleter = UpdateTimeStepsDeleter(time_steps_since_update=5)
        
        min_detections = 3  # number of detections required to begin a track
        initiator_prior_state = GaussianState(
            state_vector=np.array([[0], [0], [0], [0], [0], [0]]),
            covar=np.diag([300, 500, 300, 500, 300, 500])**2
        )
        # covar=np.diag([1e5, 500, 1e5, 500, 1e5, 500])**2
        # initiator_meas_model = CartesianToElevationBearingRange(
        #     ndim_state=6,
        #     mapping=np.array([0, 2, 4]),
        #     noise_covar=noise_covar
        # )

        self.initiator = MultiMeasurementInitiator(
            prior_state=initiator_prior_state,
            measurement_model=self.measurement_model,
            deleter=self.deleter,
            data_associator=NearestNeighbour(self.hypothesiser),
            updater=self.updater,
            min_points=min_detections,
            updates_only=True
        )
        
        # self.tracker = MultiTargetMixtureTracker(
        #     initiator=self.initiator,
        #     deleter=self.deleter,
        #     detector=sim,
        #     data_associator=data_associator,
        #     updater=self.updater
        # )

    def update(self, detections, current_time):
        
        # 1. Associate
        try:
            # print(detections)
            
            hypotheses = self.associator.associate(self.tracks, detections, current_time)
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            hypotheses = {}

        # 2. Update existing tracks
        new_tracks = set()
        associated_measurements = set()
        
        for track in self.tracks:
            # Check for update from valid associations
            updated = False
            if track in hypotheses:
                track_hyps = hypotheses[track]
                if track_hyps:
                    # Simple GNN update for demo stability with JPDA logic
                    best_hyp = max(track_hyps, key=lambda x: x.probability)
                    if best_hyp and best_hyp.measurement:
                        post_state = self.updater.update(best_hyp)
                        track.append(post_state)
                        associated_measurements.add(best_hyp.measurement)
                        updated = True
            
            # If not updated (Missed Detection), append prediction so deleter knows track is aging
            # if not updated:
            #      # Force a prediction to extend the track
            #      prediction = self.predictor.predict(track, timestamp=current_time)
            #      track.append(prediction)
            #      print(f"Track {id(track)} - Missed Detection (Appended Prediction)")
            
            if self.deleter.check_for_deletion(track):
                print(f"Deleting Track {id(track)} (Stale)")
                continue
            new_tracks.add(track)
            
        self.tracks = new_tracks
        
        # 3. Initiation
        # Filter out measurements that were associated to ANY existing track (within gate)
        # to avoid initiating new tracks on clutter or ambiguous return.
        for track in hypotheses:
             track_hyps = hypotheses[track]
             if track_hyps:
                 for hyp in track_hyps:
                     if hyp.measurement:
                         associated_measurements.add(hyp.measurement)
        
        unassociated_detections = {det for det in detections if det not in associated_measurements}
        
        # Use the MultiMeasurementInitiator (3 detections in 5 steps)
        # It handles tentative tracks internally.
        new_tracks_from_initiator = self.initiator.initiate(unassociated_detections, current_time)
        self.tracks.update(new_tracks_from_initiator)
                
        return self.tracks
