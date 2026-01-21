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
    # Setup
    start_time = datetime.now()
    duration = 20 # seconds
    dt = 0.5 # seconds
    steps = int(duration / dt)
    
    # Initialize Components
    print("Initializing Simulation...")
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
        target_manager.move_targets(current_time)
        ground_truth = target_manager.get_ground_truth()
        # print(ground_truth)
        
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
            x, vx, y, vy, z, vz = state.state_vector.flatten()
            current_truth.append((x, y, z))
        history_truth.append(current_truth)
        
        all_detections = all_detections + detections
            
        current_time += timedelta(seconds=dt)
        if i % 10 == 0:
            true_dets = sum(1 for d in detections if d.metadata.get('is_target'))
            print(f"Step {i}/{steps}: {len(detections)} Detections ({true_dets} True), {len(tracks)} Tracks")
            # plotter = CustomPlotter(Dimension.THREE)
            # plotter.plot_ground_truths(ground_truth, [0, 2, 4])
            # plotter.plot_measurements(all_detections, [0, 2, 4], measurement_model=RadarMeasurementModel(center_freq=10e9, prf=5000))
            # _ = plotter.plot_tracks(tracks, [0, 2, 4], uncertainty=False, err_freq=5)


    print("Simulation Complete. Generating Plots...")
    
    # plotter = CustomPlotter(Dimension.THREE)
    # plotter.plot_ground_truths(ground_truth, [0, 2, 4])
    # plotter.plot_measurements(all_detections, [0, 2, 4], measurement_model=RadarMeasurementModel(center_freq=10e9, prf=5000))
    # _ = plotter.plot_tracks(tracks, [0, 2, 4], uncertainty=False, err_freq=5)
    
    # Visualization
    fig = plt.figure(figsize=(15, 6))
    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("3D Target Tracking")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    
    # Plot Truth
    # Reshape history for plotting lines
    # history_truth is [step][target_idx] -> needs to be [target_id][step]
    # But targets are dynamic list objects in manager, let's grab from manager directly
    for path in target_manager.targets:
        xs = [s.state_vector[0, 0] for s in path]
        ys = [s.state_vector[2, 0] for s in path]
        zs = [s.state_vector[4, 0] for s in path]
        ax1.plot(xs, ys, zs, 'k--', label='Truth' if path == target_manager.targets[0] else "")
        ax1.scatter(xs[-1], ys[-1], zs[-1], c='k', marker='x')

    # Plot Tracks
    # Tracks come and go, so history_tracks is just a snapshot list.
    # To plot lines we'd need track IDs.
    # For now, just plot points to show consistency
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
