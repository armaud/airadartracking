from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.measures import Euclidean
from datetime import datetime, timedelta
import numpy as np
from stonesoup.types.state import State, GaussianState
from stonesoup.types.track import Track
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

def compute_ospa_metric(ground_truths, tracks, c=10, p=1):
    """
    Compute OSPA metric for multitarget tracking evaluation.
    
    Parameters:
    -----------
    ground_truths : list of GroundTruthPath
        List of ground truth trajectories
    tracks : list of Track
        List of tracker-generated tracks
    c : float, optional (default=10)
        Cut-off parameter for OSPA metric
    p : int, optional (default=1)
        Order parameter for OSPA metric (p=1 is common)
    
    Returns:
    --------
    ospa_generator : OSPAMetric
        Generator that yields OSPA metrics over time
    """
    
    # Create OSPA metric generator
    ospa_generator = OSPAMetric(
        c=c,  # Cut-off distance
        p=p,  # Order parameter
        measure=Euclidean((0, 2, 4))  # Position indices for 2D case
    )
    
    # Generate metrics
    sstracks = convert_tracks_to_stonesoup(tracks)
    ospa_metrics = ospa_generator.compute_over_time(
        measured_states=set(sstracks),
        meas_ids=1,
        truth_states=set(ground_truths),
        truth_ids=1
    )
    
    return ospa_metrics


def convert_tracks_to_stonesoup(tracks_list, start_time=None, dt=1.0):
    """
    Convert list of tracks (each track is a list of states) to Stone Soup Track objects.
    
    Parameters:
    -----------
    tracks_list : list of lists
        Each inner list represents one track
        Format options:
        1. List of [x, y] positions: [[x1, y1], [x2, y2], ...]
        2. List of [x, vx, y, vy] states: [[x1, vx1, y1, vy1], ...]
        3. List of [timestamp, x, y]: [[t1, x1, y1], [t2, x2, y2], ...]
        
    start_time : datetime, optional
        Start time for the simulation. If None, uses current time.
        
    dt : float, optional
        Time step between states (in seconds). Default is 1.0.
        Ignored if timestamps are included in the data.
    
    Returns:
    --------
    stonesoup_tracks : list of Track objects
        List of Stone Soup Track objects
    
    Example:
    --------
    # If your tracks look like:
    # tracks = [
    #     [[0, 0], [1, 1], [2, 2]],      # Track 1
    #     [[10, 10], [11, 11], [12, 12]]  # Track 2
    # ]
    
    stonesoup_tracks = convert_tracks_to_stonesoup(tracks)
    """
    
    if start_time is None:
        start_time = datetime.now()
    
    stonesoup_tracks = []
    
    for track_id, track_data in enumerate(tracks_list):
        if len(track_data) == 0:
            continue  # Skip empty tracks
            
        states = []
        
        for time_idx, state_data in enumerate(track_data):
            # Determine the format of state_data
            state_array = np.array(state_data)
            
            # Check if timestamps are included
            has_timestamp = False
            if len(state_array) in [3, 5, 7]:  # Possibly [t, x, y] or [t, x, vx, y, vy]
                # Assume first element is timestamp if it's significantly different
                # This is a heuristic - adjust based on your data
                has_timestamp = True
                timestamp = start_time + timedelta(seconds=float(state_array[0]))
                state_vector = state_array[1:]
            else:
                timestamp = start_time + timedelta(seconds=time_idx * dt)
                state_vector = state_array
            
            # Convert position-only to state vector if needed
            if len(state_vector) == 2:  # [x, y]
                # Create a 4D state vector [x, vx, y, vy]
                # Estimate velocities if possible
                if time_idx > 0 and len(states) > 0:
                    prev_state = states[-1].state_vector
                    prev_time = states[-1].timestamp
                    dt_actual = (timestamp - prev_time).total_seconds()
                    
                    vx = (state_vector[0] - prev_state[0]) / dt_actual if dt_actual > 0 else 0
                    vy = (state_vector[1] - prev_state[2]) / dt_actual if dt_actual > 0 else 0
                    state_vector = np.array([state_vector[0], vx, state_vector[1], vy])
                else:
                    # First state - no velocity information
                    state_vector = np.array([state_vector[0], 0, state_vector[1], 0])
            
            elif len(state_vector) == 3:  # [x, y, z] - 3D position
                if time_idx > 0 and len(states) > 0:
                    prev_state = states[-1].state_vector
                    prev_time = states[-1].timestamp
                    dt_actual = (timestamp - prev_time).total_seconds()
                    
                    vx = (state_vector[0] - prev_state[0]) / dt_actual if dt_actual > 0 else 0
                    vy = (state_vector[1] - prev_state[2]) / dt_actual if dt_actual > 0 else 0
                    vz = (state_vector[2] - prev_state[4]) / dt_actual if dt_actual > 0 else 0
                    state_vector = np.array([
                        state_vector[0], vx, 
                        state_vector[1], vy,
                        state_vector[2], vz
                    ])
                else:
                    state_vector = np.array([
                        state_vector[0], 0,
                        state_vector[1], 0,
                        state_vector[2], 0
                    ])
            
            # Create State object
            state = State(
                state_vector=state_vector.reshape(-1, 1),
                timestamp=timestamp
            )
            
            states.append(state)
        
        # Create Track object
        track = Track(states, id=f"track_{track_id}")
        stonesoup_tracks.append(track)
    
    return stonesoup_tracks
