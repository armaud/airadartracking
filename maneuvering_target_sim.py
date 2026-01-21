import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from scipy.stats import multivariate_normal
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, 
    ConstantVelocity, 
    KnownTurnRate, 
    SingerApproximate,
    ConstantAcceleration,
    LinearGaussianTransitionModel
)
from stonesoup.types.state import GaussianState
from stonesoup.base import Property

class TargetCategory(Enum):
    COMMERCIAL = "Commercial"
    FIGHTER = "Fighter"
    DRONE = "Drone"
    CRUISE_MISSILE = "Cruise Missile"
    BALLISTIC_MISSILE = "Ballistic Missile"

class CoordinatedTurn9D(LinearGaussianTransitionModel):
    """
    9D Coordinated Turn Model.
    Applies Coordinated Turn (CT) to X-Y dimensions (indices 0,1,3,4)
    and Constant Velocity/Acceleration logic to Z (indices 6,7,8).
    Decays Acceleration (indices 2,5,8) or treats them as noise.
    State: [x, vx, ax, y, vy, ay, z, vz, az]
    """
    turn_rate: float = Property(doc="Turn rate in rad/s")
    turn_noise_diff_coeff: float = Property(doc="Turn noise diffusion coefficient")
    accel_noise_diff_coeff: float = Property(doc="Acceleration noise diffusion coefficient")

    @property
    def ndim_state(self):
        return 9

    def matrix(self, time_interval: timedelta, **kwargs):
        dt = time_interval.total_seconds()
        omega = self.turn_rate
        
        # CT 2D Matrix elements
        if abs(omega) > 1e-6:
            sin_w = np.sin(omega * dt)
            cos_w = np.cos(omega * dt)
            mw = omega
        else:
            sin_w = 0
            cos_w = 1
            mw = 1 # Avoid div by zero, effectively CV limit

        # We construct a 9x9 matrix
        # x(0), vx(1), ax(2), y(3), vy(4), ay(5), z(6), vz(7), az(8)
        F = np.eye(9)
        
        # X-Y Coupled Turn (CV-like turn on pos/vel, ignoring ax/ay for the turn propagation to avoid conflict)
        # Position updates
        F[0, 1] = sin_w / mw
        F[3, 4] = sin_w / mw
        F[0, 4] = -(1 - cos_w) / mw
        F[3, 1] = (1 - cos_w) / mw
        
        # Velocity updates
        F[1, 1] = cos_w
        F[4, 4] = cos_w
        F[1, 4] = -sin_w
        F[4, 1] = sin_w

        # For Z, we use Constant Velocity or Constant Acceleration logic
        # Let's simple CV for Z to avoid complexity
        F[6, 7] = dt
        
        # Acceleration Decay (Simple Singer-like decay or just 1 for CA)
        # Here we let previous acceleration persist but not influence turn to keep it stable
        # or we zero it out if we assume CT is steady state turn.
        # Let's keep it as identity for now (Constant Accelerationish)
        return F

    def covar(self, time_interval: timedelta, **kwargs):
        dt = time_interval.total_seconds()
        # Simple diagonal noise for robustness
        # Higher noise on acceleration indices
        Q = np.eye(9) * 1e-9 # Base small noise
        
        q_turn = self.turn_noise_diff_coeff
        q_accel = self.accel_noise_diff_coeff
        
        # Add process noise
        # This is a simplification. Ideally we integrate process noise Q_c
        Q[1, 1] += q_turn * dt 
        Q[4, 4] += q_turn * dt
        Q[2, 2] += q_accel * dt
        Q[5, 5] += q_accel * dt
        Q[7, 7] += q_accel * dt
        Q[8, 8] += q_accel * dt
        
        return Q

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
        # 1. Commercial Aircraft Models (9D)
        # Benign Singer (approximates CV)
        alpha_comm = 0.1
        sigma_comm = 0.1
        q_comm = 2 * alpha_comm * (sigma_comm ** 2)
        singer_comm = SingerApproximate(noise_diff_coeff=q_comm, damping_coeff=alpha_comm)
        
        self.commercial_models = [
            CombinedLinearGaussianTransitionModel([singer_comm, singer_comm, singer_comm])
        ]
        self.commercial_probs = np.array([[1.0]])

        # 2. Fighter Aircraft Models (9D)
        # Logic: 0: Singer (Standard), 1: Constant Accel (Sustain), 2: Coordinated Turn (Left), 3: CT (Right)
        
        # Singer
        alpha_fight = 0.1 
        sigma_fight = 20.0 
        q_singer = 2 * alpha_fight * (sigma_fight ** 2)
        m_singer = CombinedLinearGaussianTransitionModel([
            SingerApproximate(noise_diff_coeff=q_singer, damping_coeff=alpha_fight),
            SingerApproximate(noise_diff_coeff=q_singer, damping_coeff=alpha_fight),
            SingerApproximate(noise_diff_coeff=q_singer, damping_coeff=alpha_fight)
        ])
        
        # Constant Accel (Sustain G)
        q_ca = 5.0**2
        m_ca = CombinedLinearGaussianTransitionModel([
            ConstantAcceleration(noise_diff_coeff=q_ca),
            ConstantAcceleration(noise_diff_coeff=q_ca),
            ConstantAcceleration(noise_diff_coeff=q_ca)
        ])
        
        # Coordinated Turn
        # 3 deg/s ~ 0.05 rad/s
        turn_q = 1.0 # m/s^2 (vel noise)
        acc_q = 1.0 # m/s^2 (acc noise)
        m_ct_left = CoordinatedTurn9D(turn_rate=np.radians(3), turn_noise_diff_coeff=turn_q, accel_noise_diff_coeff=acc_q)
        m_ct_right = CoordinatedTurn9D(turn_rate=np.radians(-3), turn_noise_diff_coeff=turn_q, accel_noise_diff_coeff=acc_q)

        self.fighter_models = [m_singer, m_ca, m_ct_left, m_ct_right]
        # Switching probabilities
        # [Singer, CA, Left, Right]
        self.fighter_probs = np.array([
            [0.70, 0.10, 0.10, 0.10], # From Singer (Default)
            [0.50, 0.50, 0.00, 0.00], # From CA (Return to Singer or sustain)
            [0.20, 0.00, 0.80, 0.00], # From Left (Sustain turn)
            [0.20, 0.00, 0.00, 0.80]  # From Right (Sustian turn)
        ])

        # 3. Drone Models (9D)
        # Logic: 0: Singer (Twitchy), 1: Constant Accel (Burst)
        
        # Singer (Twitchy)
        # alpha = 1.0 (tau=1s), sigma reduced to 2.0 to prevent high speed buildup
        alpha_drone = 1.0
        sigma_drone = 2.0
        q_drone_singer = 2 * alpha_drone * (sigma_drone ** 2)
        m_drone_singer = CombinedLinearGaussianTransitionModel([
             SingerApproximate(noise_diff_coeff=q_drone_singer, damping_coeff=alpha_drone),
             SingerApproximate(noise_diff_coeff=q_drone_singer, damping_coeff=alpha_drone),
             SingerApproximate(noise_diff_coeff=q_drone_singer, damping_coeff=alpha_drone)
        ])
        
        # CA (Burst)
        # Reduced noise to 2.0 m/s^2 to limit burst speed
        q_drone_ca = 2.0**2
        m_drone_ca = CombinedLinearGaussianTransitionModel([
            ConstantAcceleration(noise_diff_coeff=q_drone_ca),
            ConstantAcceleration(noise_diff_coeff=q_drone_ca),
            ConstantAcceleration(noise_diff_coeff=q_drone_ca)
        ])
        
        self.drone_models = [m_drone_singer, m_drone_ca]
        self.drone_probs = np.array([
            [0.80, 0.20], # Mostly twitchy singer
            [0.60, 0.40]  # Return from burst
        ])

        # 4. Cruise Missile Models (9D)
        # Similar to Commercial but more agile if needed, generally steady Singer
        alpha_cruise = 0.1
        sigma_cruise = 5.0
        q_cruise = 2 * alpha_cruise * (sigma_cruise ** 2)
        
        self.cruise_models = [
            CombinedLinearGaussianTransitionModel([
                SingerApproximate(noise_diff_coeff=q_cruise, damping_coeff=alpha_cruise),
                SingerApproximate(noise_diff_coeff=q_cruise, damping_coeff=alpha_cruise),
                SingerApproximate(noise_diff_coeff=q_cruise, damping_coeff=alpha_cruise)
            ])
        ]
        self.cruise_probs = np.array([[1.0]])

        # 5. Ballistic Missile Models (9D)
        # Gravity driven Constant Acceleration. 
        # Very low process noise because it follows a ballistic trajectory.
        q_ballistic = 0.1**2 
        self.ballistic_models = [
            CombinedLinearGaussianTransitionModel([
                ConstantAcceleration(noise_diff_coeff=q_ballistic),
                ConstantAcceleration(noise_diff_coeff=q_ballistic),
                ConstantAcceleration(noise_diff_coeff=q_ballistic)
            ])
        ]
        self.ballistic_probs = np.array([[1.0]])

    def _generate_targets(self):
        """
        Initialize targets with random types, positions and velocities.
        State vector: ALL 9D [x, vx, ax, y, vy, ay, z, vz, az]
        """
        for i in range(self.n_targets):
            category = np.random.choice(list(TargetCategory))
            
            # Position
            x = np.random.uniform(self.area_range[0][0], self.area_range[0][1])
            y = np.random.uniform(self.area_range[1][0], self.area_range[1][1])
            
            # Altitude
            if category == TargetCategory.DRONE:
                z = np.random.uniform(100, 1000)
            elif category == TargetCategory.CRUISE_MISSILE:
                z = np.random.uniform(100, 500) # Low altitude
            elif category == TargetCategory.BALLISTIC_MISSILE:
                z = np.random.uniform(300000, 700000) # High altitude launch/re-entry
            elif category == TargetCategory.FIGHTER:
                z = np.random.uniform(3000, 12000)
            else: # Commercial
                z = np.random.uniform(8000, 12000)

            # Speed and RCS
            if category == TargetCategory.COMMERCIAL:
                speed = np.random.uniform(200, 280) 
                rcs = np.random.uniform(10, 50)
            elif category == TargetCategory.FIGHTER:
                speed = np.random.uniform(300, 600) 
                rcs = np.random.uniform(1, 5)
            elif category == TargetCategory.DRONE:
                speed = np.random.uniform(20, 50) 
                rcs = np.random.uniform(0.1, 1)
            elif category == TargetCategory.CRUISE_MISSILE:
                speed = np.random.uniform(250, 350) 
                rcs = np.random.uniform(0.5, 2)
            else: # Ballistic Missile
                speed = np.random.uniform(800, 1500) 
                rcs = np.random.uniform(1, 10)

            # Velocity components
            heading = np.random.uniform(0, 2 * np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            vz = 0 
            az = 0
            
            if category == TargetCategory.BALLISTIC_MISSILE:
                 vz = np.random.uniform(-100, 100)
                 az = -9.81

            # Unified 9D State
            # [x, vx, ax, y, vy, ay, z, vz, az]
            state_vector = np.array([[x], [vx], [0], [y], [vy], [0], [z], [vz], [az]])

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

            # Check for Ballistic Missile impact
            if path.category == TargetCategory.BALLISTIC_MISSILE:
                z_pos = path[-1].state_vector[6, 0]
                if z_pos <= 0:
                    # Target has hit the ground, stop updating
                    continue

            # Select model based on category
            if path.category == TargetCategory.COMMERCIAL:
                models = self.commercial_models
                probs = self.commercial_probs
            elif path.category == TargetCategory.FIGHTER:
                models = self.fighter_models
                probs = self.fighter_probs
            elif path.category == TargetCategory.DRONE:
                models = self.drone_models
                probs = self.drone_probs
            elif path.category == TargetCategory.CRUISE_MISSILE:
                models = self.cruise_models
                probs = self.cruise_probs
            else: # Ballistic Missile
                models = self.ballistic_models
                probs = self.ballistic_probs

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

            # Altitude Constraint for non-Ballistic targets
            if path.category != TargetCategory.BALLISTIC_MISSILE:
                z_idx, vz_idx = 6, 7
                if new_state_vector[z_idx, 0] < 50:
                    new_state_vector[z_idx, 0] = 50.0
                    if new_state_vector[vz_idx, 0] < 0:
                        new_state_vector[vz_idx, 0] = 0.0 # Stop descent
            
            new_state = GroundTruthState(
                new_state_vector,
                timestamp=current_time
            )
            path.append(new_state)
    
    def get_ground_truth(self):
        return self.targets
