import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta

from maneuvering_target_sim import ManeuveringTargetManager
from radar_sim import RadarEmulator, RadarMeasurementModel
from tracker import TrackerManager
from stonesoup.plotter import Plotter, Dimension



class CustomPlotter(Plotter):
    def _conv_measurements(self, *args, **kwargs):
        # Get the standard output (dictionaries of tuples of arrays)
        conv_detections, conv_clutter = super()._conv_measurements(*args, **kwargs)
        
        # Helper to flatten the arrays inside the tuples
        def flatten_dict_values(d):
            new_d = {}
            for key, val in d.items():
                # val is typically (array([x]), array([y]), ...)
                # We want (x, y, ...)
                # Extract scalar if possible, or flattened array
                new_val = []
                for v in val:
                    if hasattr(v, 'item') and v.size == 1:
                        new_val.append(v.item())
                    else:
                        new_val.append(np.array(v).flatten())
                new_d[key] = tuple(new_val)
            return new_d

        return flatten_dict_values(conv_detections), flatten_dict_values(conv_clutter)


def main():
    # Args
    parser = argparse.ArgumentParser(description="Radar Tracking Simulation")
    parser.add_argument('--dataset', type=str, help="Path to ground truth pickle file", default=None)
    args = parser.parse_args()

    # Setup
    start_time = datetime.now()
    duration = 20 # seconds
    dt = 1.0 # seconds
    steps = int(duration / dt)
    
    # Initialize Components
    print("Initializing Simulation...")
    
    ground_truth_loaded = None
    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        with open(args.dataset, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'metadata' in data:
             # New format
             ground_truth_loaded = data['ground_truth']
             meta = data['metadata']
             dt = meta.get('dt', 1.0)
             duration = meta.get('duration', 20.0)
             start_time = meta.get('start_time', datetime.now())
             print(f"Loaded Metadata: dt={dt}s, Duration={duration}s")
        else:
             # Old format (list)
             ground_truth_loaded = data
             dt = 1.0 # Default fallback
        
        # Verify and extract duration/start time if possible, or just use list
        # We will iterate through available steps in loaded data
        # Assuming ground_truth_loaded is list of GroundTruthPath
        n_targets = len(ground_truth_loaded)
        print(f"Loaded {n_targets} targets.")
        
        # Override steps based on data length roughly
        # If metadata provided duration, calculate steps. Else rely on data length.
        if isinstance(data, dict) and 'metadata' in data:
            steps = int(duration / dt)
        else:
            steps = len(ground_truth_loaded[0])
            start_time = ground_truth_loaded[0][0].timestamp
        
        # Mock manager or just bypass
        target_manager = None
    else:
        target_manager = ManeuveringTargetManager(start_time, n_targets=2)
    # Using default radar parameters (X-band ~10GHz)
    radar = RadarEmulator(center_freq=10e9, noise_power=1.0)
    tracker = TrackerManager()
    
    # History for plotting
    history_tracks = [] # List of list of points
    history_truth = []
    all_detections = []
    
    # Main Loop
    current_time = start_time
    print(f"Starting Simulation for {steps} steps...")
    
    last_rd_map = None
    
    for i in range(steps):
        # 1. Targets
        if target_manager:
            target_manager.move_targets(current_time)
            ground_truth = target_manager.get_ground_truth()
        else:
            # Dataset loaded mode
            # We need to extract the state at current step 'i' from the tracks
            # Assuming all tracks have same length and timestamps corresponding to steps
            ground_truth_step = []
            for path in ground_truth_loaded:
                if i < len(path):
                    # We might need to construct a mini-path or just pass the state
                    # radar_sim expects a list of paths or state that has .state_vector
                    # Actually radar_sim._get_target_state takes 'target' which is a path or state?
                    # It calls target[-1] or target.state_at.
                    # Let's create a temporary path with just the current state to satisfy radar_sim interface
                    current_path = [path[i]]
                    ground_truth_step.append(current_path)
            ground_truth = ground_truth_step
            # Update current time from truth
            if ground_truth:
                current_time = ground_truth[0][-1].timestamp

        # 2. Radar
        detections, rd_map = radar.get_detections(ground_truth, current_time)
        last_rd_map = rd_map
        
        # 3. Tracker
        tracks = tracker.update(detections, current_time)
        
        # Store for viz
        current_tracks = []
        for track in tracks:
            state = track.state
            x, vx, y, vy, z, vz = state.state_vector.flatten()
            current_tracks.append((x, y, z))
        history_tracks.append(current_tracks)
        
        current_truth = []
        for path in ground_truth:
            state = path[-1]
            sv = state.state_vector
            if sv.shape[0] == 9: # 9D
                 x, y, z = sv[0, 0], sv[3, 0], sv[6, 0]
            else: # 6D
                 x, y, z = sv[0, 0], sv[2, 0], sv[4, 0]
            current_truth.append((x, y, z))
        history_truth.append(current_truth)
        
        all_detections = all_detections + detections
            
        if target_manager:
            current_time += timedelta(seconds=dt)

        if i % 10 == 0:
            true_dets = sum(1 for d in detections if d.metadata.get('is_target'))
            print(f"Step {i}/{steps}: {len(detections)} Detections ({true_dets} True), {len(tracks)} Tracks")


    print("Simulation Complete. Generating Plots...")
    
    # Visualization
    fig = plt.figure(figsize=(15, 6))
    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("3D Target Tracking")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    
    # Plot Truth
    if target_manager:
        truth_source = target_manager.targets
    else:
        truth_source = ground_truth_loaded

    for path in truth_source:
        sv = path[0].state_vector
        if sv.shape[0] == 9:
             xs = [s.state_vector[0, 0] for s in path]
             ys = [s.state_vector[3, 0] for s in path]
             zs = [s.state_vector[6, 0] for s in path]
        else:
             xs = [s.state_vector[0, 0] for s in path]
             ys = [s.state_vector[2, 0] for s in path]
             zs = [s.state_vector[4, 0] for s in path]
        
        ax1.plot(xs, ys, zs, 'k--', label='Truth' if path == truth_source[0] else "")
        ax1.scatter(xs[-1], ys[-1], zs[-1], c='k', marker='x')

    # Plot Tracks
    for step_tracks in history_tracks:
        if not step_tracks: continue
        xs, ys, zs = zip(*step_tracks)
        ax1.scatter(xs, ys, zs, c='b', s=5, alpha=0.5)
        
    # Plot final track positions
    if history_tracks[-1]:
         xf, yf, zf = zip(*history_tracks[-1])
         ax1.scatter(xf, yf, zf, c='r', marker='o', label='Track Estimate')

    ax1.legend()
    
    # Range-Doppler Map (Last Frame)
    # ax2 = fig.add_subplot(122)
    # ax2.set_title("Range-Doppler Map (Last Frame)")
    # # Log scale for power
    # rd_db = 10 * np.log10(last_rd_map + 1e-10)
    # img = ax2.imshow(rd_db, aspect='auto', origin='lower', cmap='jet')
    # plt.colorbar(img, ax=ax2, label='Power (dB)')
    # ax2.set_xlabel("Doppler Bin")
    # ax2.set_ylabel("Range Bin")
    
    plt.tight_layout()
    plt.savefig('simulation_result.png')
    print("Plot saved to simulation_result.png")
    # plt.show() # Can't show in headless, relies on save

if __name__ == "__main__":
    main()
