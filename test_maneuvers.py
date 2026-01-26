from datetime import datetime, timedelta
import pickle
import numpy as np
from maneuvering_target_sim import ManeuveringTargetManager, TargetCategory

def test_maneuvers():
    start_time = datetime.now()
    manager = ManeuveringTargetManager(start_time, n_targets=1)
    
    print("\nInitial States:")
    for i, target in enumerate(manager.targets):
        state = target[0].state_vector
        # 9D: x, vx, ax, y, vy, ay, z, vz, az
        vel = np.array([state[1, 0], state[4, 0], state[7, 0]])
            
        speed =  np.linalg.norm(vel)
        print(f"Target {i} ({target.category.value}): Speed={speed.item():.2f} m/s, RCS={target.rcs:.2f}, Dim={state.shape[0]}")

    # Simulation Params
    duration = 120 # seconds
    dt = 1.0 # seconds
    
    # Update manager timestep if needed, though it defaults to 1s
    manager.timestep = timedelta(seconds=dt)
    
    print(f"\nSimulating {duration} seconds with dt={dt}...")
    current_time = start_time
    steps = int(duration / dt)
    for step in range(steps):
        current_time = start_time + (step + 1) * manager.timestep
        manager.move_targets(current_time)
        
    print("\nFinal States:")
    for i, target in enumerate(manager.targets):
        state_vec = target[-1].state_vector
        initial_vec = target[0].state_vector
        
        # Position extraction (9D simplified)
        pos = np.array([state_vec[0, 0], state_vec[3, 0], state_vec[6, 0]])
        vel = np.array([state_vec[1, 0], state_vec[4, 0], state_vec[7, 0]])
        initial_pos = np.array([initial_vec[0, 0], initial_vec[3, 0], initial_vec[6, 0]])

        speed = np.linalg.norm(vel)
        # Check if it moved
        dist = np.linalg.norm(pos - initial_pos)
        print(f"Target {i}: Speed={speed.item():.2f} m/s, Dist Moved={dist.item():.2f} m")

    # Plotting
    import matplotlib.pyplot as plt
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        pass # Recent matplotlib versions don't need this explicit import usually

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Target Trajectories")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    colors = {
        TargetCategory.COMMERCIAL: 'blue', 
        TargetCategory.FIGHTER: 'red', 
        TargetCategory.DRONE: 'green',
        TargetCategory.CRUISE_MISSILE: 'purple',
        TargetCategory.BALLISTIC_MISSILE: 'orange'
    }

    for path in manager.targets:
        xs, ys, zs = [], [], []
        
        for s in path:
            sv = s.state_vector
            xs.append(sv[0, 0])
            ys.append(sv[3, 0])
            zs.append(sv[6, 0])
        
        c = colors.get(path.category, 'black')
        label = path.category.value if path.category.value not in [l.get_label() for l in ax.get_lines()] else ""
        
        ax.plot(xs, ys, zs, color=c, label=label)
        ax.scatter(xs[-1], ys[-1], zs[-1], color=c, marker='x') # End point
        ax.scatter(xs[0], ys[0], zs[0], color=c, marker='o') # Start point

    ax.legend()
    plt.tight_layout()
    
    # Generate timestamp for both files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Plot
    plot_filename = f'dataset/maneuver_test_{timestamp}.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # Save tracks to file
    pkl_filename = f'dataset/ground_truth_{timestamp}.pkl'
    
    data = {
        'metadata': {
            'duration': duration,
            'dt': dt,
            'start_time': start_time,
            'n_targets': manager.n_targets
        },
        'ground_truth': manager.targets
    }
    
    with open(pkl_filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Ground truth tracks saved to {pkl_filename}")

if __name__ == "__main__":
    test_maneuvers()
