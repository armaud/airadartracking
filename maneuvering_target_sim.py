import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, 
    ConstantVelocity, 
    KnownTurnRate, 
    SingerApproximate,
    ConstantAcceleration
)
from stonesoup.types.state import GaussianState

class TargetCategory(Enum):
    COMMERCIAL = "Commercial"
    FIGHTER = "Fighter"
    DRONE = "Drone"

class ManeuveringTargetManager:
    """
    Manages the simulation of multiple targets in 3D space with category-specific behaviors.
    """
    def __init__(self, start_time: datetime, n_targets: int = 3, area_range: tuple = ((-10000, 10000), (-10000, 10000), (5000, 10000))):
        self.start_time = start_time
        self.n_targets = n_targets
        self.area_range = area_range
        self.targets = []
        
        # Define simulation parameters
        self.timestep = timedelta(seconds=1)
        
        self._init_models()
        self._generate_targets()

    def _init_models(self):
        # 1. Commercial Aircraft Models
        # High speed, mostly straight, slow turns (6D State)
        self.commercial_models = [
            CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1), ConstantVelocity(0.1), ConstantVelocity(0.1)]), # Straight
            CombinedLinearGaussianTransitionModel([KnownTurnRate([0.1, 0.1], np.radians(1)), ConstantVelocity(0.1)]), # Turn Left (Slow)
            CombinedLinearGaussianTransitionModel([KnownTurnRate([0.1, 0.1], np.radians(-1)), ConstantVelocity(0.1)]) # Turn Right (Slow)
        ]
        self.commercial_probs = np.array([
            [0.90, 0.05, 0.05], # From Straight
            [0.20, 0.80, 0.00], # From Left
            [0.20, 0.00, 0.80]  # From Right
        ])

        # 2. Fighter Aircraft Models (Singer Model)
        # Correlated Acceleration, smooth but agile maneuvers (9D State)
        # Singer params: damping_coeff (alpha), noise_diff_coeff (q)
        # alpha = 1/tau (maneuver time constant). Fighters tau ~ 10-20s. alpha ~ 0.05 - 0.1
        # q = 2 * alpha * sigma^2 (where sigma is maneuver std dev)
        alpha = 0.1 
        sigma = 10.0 # m/s^2 maneuverability
        q_singer = 2 * alpha * (sigma ** 2)
        
        singer_1d = SingerApproximate(noise_diff_coeff=q_singer, damping_coeff=alpha)
        
        self.fighter_models = [
            # 3D Singer Model
            CombinedLinearGaussianTransitionModel([singer_1d, singer_1d, singer_1d])
        ]
        self.fighter_probs = np.array([[1.0]]) # Single model handling the dynamics via Singer process

        # 3. Drone Models (Jerk / High Agility)
        # Low inertia, rapid acceleration changes (9D State)
        # Using Singer with high damping (short correlation time) to simulate "twitchy" behavior
        # without unbounded velocity growth.
        # alpha = 1.0 (tau = 1s), sigma = 5.0 m/s^2
        alpha_drone = 1.0
        sigma_drone = 5.0
        q_drone = 2 * alpha_drone * (sigma_drone ** 2)
        
        singer_drone = SingerApproximate(noise_diff_coeff=q_drone, damping_coeff=alpha_drone)
        
        self.drone_models = [
             CombinedLinearGaussianTransitionModel([singer_drone, singer_drone, singer_drone])
        ]
        self.drone_probs = np.array([[1.0]])

    def _generate_targets(self):
        """
        Initialize targets with random types, positions and velocities.
        State vector: 
        - Commercial (6D): [x, vx, y, vy, z, vz]
        - Fighter/Drone (9D): [x, vx, ax, y, vy, ay, z, vz, az]
        """
        for i in range(self.n_targets):
            category = np.random.choice(list(TargetCategory))
            
            # Position
            x = np.random.uniform(self.area_range[0][0], self.area_range[0][1])
            y = np.random.uniform(self.area_range[1][0], self.area_range[1][1])
            
            # Altitude
            if category == TargetCategory.DRONE:
                z = np.random.uniform(100, 1000)
            else:
                z = np.random.uniform(3000, 12000)

            # Speed and RCS
            if category == TargetCategory.COMMERCIAL:
                speed = np.random.uniform(200, 280) 
                rcs = np.random.uniform(10, 50)
            elif category == TargetCategory.FIGHTER:
                speed = np.random.uniform(300, 600) 
                rcs = np.random.uniform(1, 5)
            else: # Drone
                speed = np.random.uniform(20, 50) 
                rcs = np.random.uniform(0.1, 1)

            # Velocity components
            heading = np.random.uniform(0, 2 * np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            vz = 0 

            # Create State Vector based on Category
            if category == TargetCategory.COMMERCIAL:
                state_vector = np.array([[x], [vx], [y], [vy], [z], [vz]])
            else:
                # 9D State for Singer/CA: [x, vx, ax, y, vy, ay, z, vz, az]
                # Initialize acceleration to 0
                state_vector = np.array([[x], [vx], [0], [y], [vy], [0], [z], [vz], [0]])

            initial_state = GroundTruthState(
                state_vector,
                timestamp=self.start_time
            )
            
            path = GroundTruthPath([initial_state], id=str(i))
            path.category = category
            path.rcs = rcs
            path.current_model_index = 0 
            
            self.targets.append(path)
            print(f"Generated Target {i}: {category.value}, Speed={speed:.1f} m/s, RCS={rcs:.1f}, StateDim={len(state_vector)}")

    def move_targets(self, current_time: datetime):
        """
        Propagate targets to the current_time using their specific transition models.
        """
        for path in self.targets:
            previous_state = path[-1]
            dt = (current_time - previous_state.timestamp).total_seconds()
            
            if dt <= 0:
                continue

            # Select model based on category
            if path.category == TargetCategory.COMMERCIAL:
                models = self.commercial_models
                probs = self.commercial_probs
            elif path.category == TargetCategory.FIGHTER:
                models = self.fighter_models
                probs = self.fighter_probs
            else: # Drone
                models = self.drone_models
                probs = self.drone_probs

            # Determine next model index
            current_idx = path.current_model_index
            next_idx = np.random.choice(len(models), p=probs[current_idx])
            path.current_model_index = next_idx
            
            selected_model = models[next_idx]

            # Propagate
            new_state_vector = selected_model.function(
                previous_state, 
                noise=True, 
                time_interval=timedelta(seconds=dt)
            )
            
            new_state = GroundTruthState(
                new_state_vector,
                timestamp=current_time
            )
            path.append(new_state)
    
    def get_ground_truth(self):
        return self.targets
