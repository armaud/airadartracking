import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.state import GaussianState

class TargetManager:
    """
    Manages the simulation of multiple targets in 3D space.
    """
    def __init__(self, start_time: datetime, n_targets: int = 3, area_range: tuple = ((-1000, 1000), (-1000, 1000), (1000, 5000))):
        self.start_time = start_time
        self.n_targets = n_targets
        self.targets = []
        self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05), ConstantVelocity(0.05), ConstantVelocity(0.05)])
        self.area_range = area_range
        self._generate_targets()

    def _generate_targets(self):
        """
        Initialize targets with random positions and velocities.
        State vector: [x, vx, y, vy, z, vz]
        """
        for i in range(self.n_targets):
            x = np.random.uniform(self.area_range[0][0], self.area_range[0][1])
            y = np.random.uniform(self.area_range[1][0], self.area_range[1][1])
            z = np.random.uniform(self.area_range[2][0], self.area_range[2][1])
            
            # Random velocities between -50 and 50 m/s
            vx = np.random.uniform(-500, 500)
            vy = np.random.uniform(-500, 500)
            vz = np.random.uniform(-100, 100) # Lower vertical velocity

            # vx = 200
            # vy = 200
            # vz = 5
            initial_state = GroundTruthState(
                [x, vx, y, vy, z, vz],
                timestamp=self.start_time
            )
            
            path = GroundTruthPath([initial_state], id=str(i))
            # Assign random RCS between 0.5 and 10 m^2
            path.rcs = np.random.uniform(0.5, 10.0)
            self.targets.append(path)

    def move_targets(self, current_time: datetime):
        """
        Propagate targets to the current_time using the transition model.
        """
        for path in self.targets:
            previous_state = path[-1]
            # Time delta is handled by the transition model if we pass specific time intervals, 
            # but Stonesoup transition models often take time_interval or we just use the predictor logic.
            # However, for Ground Truth generation, we can simply sample from the model.
            
            # Calculate dt
            dt = current_time - previous_state.timestamp
            
            # Propagate
            # For simple Ground Truth, we can just use the matrix form if we want deterministic or add noise.
            # Here we use the model's function which adds process noise.
            new_state_vector = self.transition_model.function(
                previous_state, 
                noise=True, 
                time_interval=dt
            )
            
            new_state = GroundTruthState(
                new_state_vector,
                timestamp=current_time
            )
            path.append(new_state)

    def get_ground_truth(self):
        """
        Return the list of GroundTruthPaths (which contain the history).
        """
        return self.targets
