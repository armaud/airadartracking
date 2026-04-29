import torch
import pickle
import os
import random
# from Simulations.Radar.parameters_radar import h, n, m
from radar_detections import RadarEmulator
from datetime import datetime
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


m = 6 # State dimension: [px, vx, py, vy, pz, vz]
n = 3 # Observation dimension: [elevation, bearing, range]

# This file converts our Stonesoup dataset (saved as pkl) into Pytorch dataset

def convert_stonesoup_dataset(data_dir):
    all_input = []
    all_target = []
        
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} files in {data_dir}")
    
    for filename in pkl_files:
        # Setup
        start_time = datetime.now()
        duration = 20 # seconds
        dt = 1.0 # seconds
        steps = int(duration / dt)
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f_in:
            data = pickle.load(f_in)
            
        # ground_truth = data['ground_truth'] # List of GroundTruthPath
        # duration = data['metadata']['duration']
        ground_truth = data['ground_truth']
        meta = data['metadata']
        dt = meta.get('dt', 1.0)
        duration = meta.get('duration', 20.0)
        start_time = meta.get('start_time', datetime.now())
        # print(f"Loaded Metadata: dt={dt}s, Duration={duration}s")
        steps = int(duration / dt)
        radar = RadarEmulator(start_time=start_time,center_freq=10e9, noise_power=1.0)
        
        for path in ground_truth:
            # If path is 9D, convert it to 6D for the radar sensor (which expects 6D x,vx,y,vy,z,vz)
            if path[0].state_vector.shape[0] == 9:
                path_for_radar = convert_9d_to_6d_path(path)
            else:
                path_for_radar = path
                
            # Extract states for target tensor ([6, T])
            states = get_path_states(path)
            
            target_tensor = torch.stack(states).t() # [6, T]
            # Generate Measurements using the 6D path
            input_tensor = get_measurements(radar, path_for_radar)
            
            all_input.append(input_tensor)
            all_target.append(target_tensor)

    # Convert to big tensors [Batch, dim, T]
    dataset_input = torch.stack(all_input)
    dataset_target = torch.stack(all_target)
    return dataset_input, dataset_target
    # N = dataset_input.shape[0]
    # print(f"Total trajectories extracted: {N}")
    
    # # Shuffle and Split
    # indices = list(range(N))
    # random.shuffle(indices)
    
    # n_train = int(0.8 * N)
    # n_cv = int(0.1 * N)
    
    # train_idx = indices[:n_train]
    # cv_idx = indices[n_train:n_train+n_cv]
    # test_idx = indices[n_train+n_cv:]
    
    # train_input = dataset_input[train_idx]
    # train_target = dataset_target[train_idx]
    # cv_input = dataset_input[cv_idx]
    # cv_target = dataset_target[cv_idx]
    # test_input = dataset_input[test_idx]
    # test_target = dataset_target[test_idx]
    
    # print(f"Splits: Train={len(train_idx)}, CV={len(cv_idx)}, Test={len(test_idx)}")
    
    # # Save in the format expected by main_radar_comparison.py
    # torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target], output_file)
    # print(f"Dataset saved to {output_file}")
    
def convert_9d_to_6d_path(path_9d):
    """
    Converts a 9D GroundTruthPath [x, vx, ax, y, vy, ay, z, vz, az] 
    into a 6D GroundTruthPath [x, vx, y, vy, z, vz].
    """
    states_6d = []
    indices = [0, 1, 3, 4, 6, 7]
    for state_9d in path_9d:
        state_vector_6d = state_9d.state_vector[indices, :]
        state_6d = GroundTruthState(state_vector_6d, timestamp=state_9d.timestamp)
        states_6d.append(state_6d)
    return GroundTruthPath(states_6d, id=path_9d.id)

def get_measurements(radar, path):
    m_states = []
    all_measurements = radar.get_detections(path, datetime.now())
    # Sort measurements by timestamp to align with sequential states
    sorted_measurements = sorted(all_measurements, key=lambda x: x.timestamp)
    for meas in sorted_measurements:
        v = torch.from_numpy(meas.state_vector.astype(float)).float().view(n)
        m_states.append(v)
    return torch.stack(m_states).t()
    # return m_states
    
def get_path_states(path):
    m_states = []
    for state in path.states:
        sv = state.state_vector
        # Check if we have 9D [x, vx, ax, y, vy, ay, z, vz, az] 
        # or 6D [x, vx, y, vy, z, vz]
        if sv.shape[0] == 9:
            indices = [0, 1, 3, 4, 6, 7]
        elif sv.shape[0] == 6:
            indices = [0, 1, 2, 3, 4, 5]
        else:
            raise ValueError(f"Unexpected state dimension: {sv.shape[0]}. Expected 6 or 9.")
            
        full_state = sv[indices, :].astype(float)
        v = torch.from_numpy(full_state).float().view(m)
        
        if v.shape[0] != m:
            raise ValueError(f"Extracted state size {v.shape[0]} does not match expected size m={m}")
            
        m_states.append(v)
    return m_states

if __name__ == "__main__":
    TRAIN_DATA_DIR = r'C:\Users\Admin\projects\radartracking\dataset\Train'
    TEST_DATA_DIR = r'C:\Users\Admin\projects\radartracking\dataset\Test'
    OUTPUT_FILE = r'C:\Users\Admin\projects\radartracking\dataset\torch_data.pt'
    
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))
        
    data_input, data_target = convert_stonesoup_dataset(TRAIN_DATA_DIR)
    test_input, test_target = convert_stonesoup_dataset(TEST_DATA_DIR)
    
    # Divide training samples into training/ validation
    N = data_input.shape[0]
    n_train = int(0.9 * N)
    n_cv = int(0.1 * N)
    # Shuffle and Split
    indices = list(range(N))
    random.shuffle(indices)
    
    train_idx = indices[:n_train]
    cv_idx = indices[n_train:n_train+n_cv]
    
    train_input = data_input[train_idx]
    train_target = data_target[train_idx]
    cv_input = data_input[cv_idx]
    cv_target = data_target[cv_idx]
    print(f"Splits: Train={len(train_idx)}, CV={len(cv_idx)}, Test={test_input.shape[0]}")
    torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target], OUTPUT_FILE)    
    print(f"Dataset saved to {OUTPUT_FILE}")