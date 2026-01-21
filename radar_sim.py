import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.base import Property

from stonesoup.models.base import ReversibleModel

class RadarMeasurementModel(MeasurementModel, ReversibleModel):
    """
    Non-linear measurement model for [Range, Azimuth, Elevation, RangeRate]
    State: [x, vx, y, vy, z, vz]
    """
    center_freq: float = Property(default=10e9, doc="Radar center frequency")
    prf: float = Property(default=5000, doc="Pulse Repetition Frequency")
    
    #    out = np.random.multivariate_normal(np.zeros(self.ndim_meas), self.noise_covar, num_samples).T
    # But wait, we need to fix the class definition first.
    
    def __init__(self, ndim_state=6, mapping=(0, 2, 4), center_freq=10e9, prf=5000):
        super().__init__(ndim_state=ndim_state, mapping=mapping)
        self.noise_covar = np.diag([10**2, np.radians(1)**2, np.radians(1)**2, 5**2]) # Widen velocity gate
        self.c = 3e8
        self.lambda_w = self.c / self.center_freq
        self.v_ambig = self.prf * self.lambda_w / 2.0

    def covar(self, **kwargs):
        return self.noise_covar
        
    def function(self, state, noise=False, **kwargs):
        x = state.state_vector[0, 0]
        y = state.state_vector[2, 0]
        z = state.state_vector[4, 0]
        vx = state.state_vector[1, 0]
        vy = state.state_vector[3, 0]
        vz = state.state_vector[5, 0]

        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.arctan2(y, x)
        el = np.arcsin(z / r) if r != 0 else 0
        
        # Range Rate
        if r == 0:
            rr = 0
        else:
            rr = (x*vx + y*vy + z*vz) / r
            
        # Aliasing (Wrapping)
        # We need to wrap the predicted range rate into the interval [-v_ambig/2, v_ambig/2]
        # This assumes the measurement is always aliased this way
        
        rr_wrapped = rr - self.v_ambig * round(rr / self.v_ambig)

        out = np.array([r, az, el, rr_wrapped]).reshape(4, 1)
        
        if noise:
            out += self.rvs()
            
        return out

    def jacobian(self, state, **kwargs):
        # Note: Jacobian technically ignores wrapping discontinuities, which is standard for EKF
        # unless we are right on the edge.
        x = state.state_vector[0, 0]
        y = state.state_vector[2, 0]
        z = state.state_vector[4, 0]
        vx = state.state_vector[1, 0]
        vy = state.state_vector[3, 0]
        vz = state.state_vector[5, 0]
        
        r2 = x**2 + y**2 + z**2
        r = np.sqrt(r2)
        r3 = r2 * r
        
        if r == 0: return np.zeros((4, 6))

        xy2 = x**2 + y**2
        
        # Row 0: R
        dr_dx = x/r
        dr_dy = y/r
        dr_dz = z/r
        
        # Row 1: Az
        if xy2 == 0:
            daz_dx, daz_dy = 0, 0
        else:
            daz_dx = -y / xy2
            daz_dy = x / xy2
        
        # Row 2: El
        pre = r / np.sqrt(xy2) if xy2 > 0 else 0
        del_dx = pre * (-z*x/r3)
        del_dy = pre * (-z*y/r3)
        del_dz = pre * ((x**2+y**2)/r3)
        
        # Row 3: RR
        drr_dx = (vx*r - ((x*vx + y*vy + z*vz)*x/r))/r2
        drr_dy = (vy*r - ((x*vx + y*vy + z*vz)*y/r))/r2
        drr_dz = (vz*r - ((x*vx + y*vy + z*vz)*z/r))/r2
        
        drr_dvx = x/r
        drr_dvy = y/r
        drr_dvz = z/r
        
        H = np.zeros((4, 6))
        H[0, 0] = dr_dx; H[0, 2] = dr_dy; H[0, 4] = dr_dz
        H[1, 0] = daz_dx; H[1, 2] = daz_dy
        H[2, 0] = del_dx; H[2, 2] = del_dy; H[2, 4] = del_dz
        H[3, 0] = drr_dx; H[3, 1] = drr_dvx; H[3, 2] = drr_dy; H[3, 3] = drr_dvy; H[3, 4] = drr_dz; H[3, 5] = drr_dvz
        
        return H

    def rvs(self, num_samples=1, **kwargs):
        out = np.random.multivariate_normal(np.zeros(self.ndim_meas), self.noise_covar, num_samples).T
        if num_samples == 1:
            return out[:, 0]
        return out

    def pdf(self, state1, state2, **kwargs):
        return super().pdf(state1, state2, **kwargs)
    
    def inverse_function(self, detection, **kwargs):
        r = detection.state_vector[0]
        az = detection.state_vector[1]
        el = detection.state_vector[2]
        rr = detection.state_vector[3]
        # Pre-calculate trig
        cos_el = np.cos(el)
        sin_el = np.sin(el)
        cos_az = np.cos(az)
        sin_az = np.sin(az)

        # Position Mapping
        x = r * cos_el * cos_az
        y = r * cos_el * sin_az
        z = r * sin_el

        # Velocity Mapping (Radial projection)
        # Since we lack angular rates, we project the radial velocity 
        # along the line-of-sight vector.
        vx = rr * cos_el * cos_az
        vy = rr * cos_el * sin_az
        vz = rr * sin_el

        # Return in the interleaved format: [x, vx, y, vy, z, vz]
        out = np.array([[x], [vx], [y], [vy], [z], [vz]])
        return out
        
    
    @property
    def ndim_meas(self):
        return 4


class RadarEmulator:
    """
    Simulates a Pulse Doppler Radar with Range-Doppler map generation and CFAR detection.
    """
    def __init__(self, center_freq=10e9, bandwidth=5e6, prf=5000, 
                 n_pulses=64, noise_power=1.0, probability_false_alarm=1e-4,
                 transmit_power=10000.0, antenna_gain_db=30.0):
        self.fc = center_freq
        self.c = 3e8
        self.bw = bandwidth
        self.prf = prf
        self.n = n_pulses
        self.noise_power = noise_power # Expected to be unitless or relative noise floor
        self.pfa = probability_false_alarm
        self.pt = transmit_power
        self.gt_db = antenna_gain_db
        self.gt = 10**(self.gt_db/10)
        
        # Resolution
        self.delta_r = self.c / (2 * self.bw)
        self.delta_v = (self.c * self.prf) / (2 * self.fc * self.n)
        self.max_range = self.c / (2 * self.prf) # Unambiguous range (simple) 
        # Actually standard equation is c / (2 * PRF) but let's stick to a window
        self.max_range = 50000 # fixed window for sim
        
        self.n_range_bins = int(self.max_range / self.delta_r)
        self.n_doppler_bins = self.n

    def _get_target_state(self, target, current_time):
        # Extract R, Az, El, V_r from target
        # Assuming target state is [x, vx, y, vy, z, vz]
        # Interpolation might be needed if exact time not matches, 
        # but here we assume steps match or we take latest.
        # state = target.state_at(current_time)
        state = target[-1]
        
        x, vx, y, vy, z, vz = state.state_vector.flatten()
        
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Radial velocity: v dot r_hat
        r_vec = np.array([x, y, z])
        v_vec = np.array([vx, vy, vz])
        vr = np.dot(v_vec, r_vec) / r
        
        az = np.arctan2(y, x)
        el = np.arcsin(z / r)
        
        return r, az, el, vr

    def calculate_snr(self, target_rcs, target_range):
        """
        Calculate SNR using Radar Range Equation.
        SNR = (Pt * G^2 * lambda^2 * sigma) / ((4pi)^3 * R^4 * Noise)
        """
        if target_range <= 0:
            return 0
            
        wavelength = self.c / self.fc
        
        # Numerator
        num = self.pt * (self.gt**2) * (wavelength**2) * target_rcs
        
        # Denominator
        # We need a physical noise power to make this meaningful.
        # Let's assume standard thermal noise kTB * F
        # k = 1.38e-23, T = 290K
        # If simulation noise_power is 1, we need to map physical SNR to this relative scale
        # Let's calculate physical Noise Power (Pn)
        k = 1.38e-23
        t = 290
        nf_db = 3 # 3dB Noise Figure
        nf = 10**(nf_db/10)
        pn = k * t * self.bw * nf
        
        denom = ((4 * np.pi)**3) * (target_range**4) * pn
        
        snr_linear = num / denom
        return snr_linear

    def generate_rd_map(self, targets, current_time):
        """
        Generates a Range-Doppler map with targets and noise.
        """
        # Initialize map with complex noise
        # magnitude^2 is exponential, real/imag are gaussian
        noise_real = np.random.normal(0, np.sqrt(self.noise_power/2), (self.n_range_bins, self.n_doppler_bins))
        noise_imag = np.random.normal(0, np.sqrt(self.noise_power/2), (self.n_range_bins, self.n_doppler_bins))
        rd_map_complex = noise_real + 1j * noise_imag

        ########################################
        # rd_map_complex = 0*rd_map_complex
        #######################################
        for target in targets:
            r, _, _, vr = self._get_target_state(target, current_time)
            
            # Get RCS
            rcs = getattr(target, 'rcs', 1.0) # Default 1.0 if not present
            
            # Calculate SNR
            snr_linear = self.calculate_snr(rcs, r)
            
            # Signal Amplitude
            # Signal Power Ps = SNR * NoisePower
            # Amplitude = sqrt(Ps)
            amplitude = np.sqrt(snr_linear * self.noise_power)
            
            # Find bins
            r_bin = int(r / self.delta_r)
            
            # Doppler bin: shift 0 velocity to center? 
            # Or standard FFT output 0..Fs
            # Max unambiguous velocity = lambda * PRF / 2
            # v_bin index corresponds to frequencies -PRF/2 to PRF/2 usually
            # Let's map directly
            lambda_w = self.c / self.fc
            doppler_freq = 2 * vr / lambda_w
            # Map freq to bin (0 to PRF) or (-PRF/2 to PRF/2)
            # Bin index = (freq / PRF) * N
            # Wrap around
            v_bin = int((doppler_freq / self.prf) * self.n_doppler_bins) % self.n_doppler_bins
            
            if 0 <= r_bin < self.n_range_bins:
                # Add signal
                # For simplicity, add to single bin. Could spread (sinc).
                rd_map_complex[r_bin, v_bin] += amplitude
                
        return np.abs(rd_map_complex)**2 # Return Power

    def apply_cfar(self, rd_map):
        """
        CA-CFAR implementation 1D (per Doppler slice) or 2D.
        Let's do CA-CFAR 1D along Range for each Doppler bin for simplicity/speed.
        """
        detections = []
        rows, cols = rd_map.shape
        
        # CFAR parameters
        guard_cells = 2
        ref_cells = 10
        alpha = self.n * (self.pfa**(-1/self.n) - 1) # This is for square law detector? 
        # Standard CA-CFAR alpha for exponential distribution
        # alpha = N (P_fa ^ (-1/N) - 1) where N is ref cells
        N_ref = 2 * ref_cells
        alpha = N_ref * (self.pfa**(-1/N_ref) - 1)

        # Iterate
        # Simplified: Just thresholding the map based on local average
        # We'll use a convolution approach for speed if using 2D, but loops are fine for simulation
        
        hit_indices = []
        
        for j in range(cols): # Loop doppler
            for i in range(rows): # Loop range
                if i < ref_cells + guard_cells or i > rows - ref_cells - guard_cells - 1:
                    continue
                
                # Training cells (Range dimension only)
                train_window = np.concatenate((rd_map[i - guard_cells - ref_cells : i - guard_cells, j],
                                             rd_map[i + guard_cells + 1 : i + guard_cells + ref_cells + 1, j]))
                
                noise_est = np.mean(train_window)
                threshold = alpha * noise_est
                
                if rd_map[i, j] > threshold:
                    hit_indices.append((i, j))
        
        return hit_indices

    def get_detections(self, targets, current_time):
        rd_map = self.generate_rd_map(targets, current_time)
        hits = self.apply_cfar(rd_map)
        
        detections = []
        for r_bin, v_bin in hits:
            # Convert back to physical measurements
            r_meas = r_bin * self.delta_r
            
            # Velocity wrapping handling for display/tracking
            # v = (bin / N) * PRF * lambda / 2
            lambda_w = self.c / self.fc
            
            # Handle wrapping: if bin > N/2, it's negative velocity
            if v_bin > self.n_doppler_bins / 2:
                v_freq = (v_bin - self.n_doppler_bins) * (self.prf / self.n_doppler_bins)
            else:
                v_freq = v_bin * (self.prf / self.n_doppler_bins)
                
            v_meas = v_freq * lambda_w / 2
            
            # Since CFAR is only R/Rdot, we need Az/El.
            # In a real radar, beamforming gives Az/El estimates.
            # Here we cheat slightly: We find the CLOSEST target to this (R, Rdot) detection
            # and use its Az/El + noise. Or we simulate 'Scanning' and only return hits 
            # if the beam is pointing there.
            # Simplest for Stonesoup demo:
            # Associate hit with ground truth to 'fake' the angle measurement or 
            # just search for all targets that could match this bin.
            
            # Let's add independent noise for angle to simulate measurement uncertainty
            # If multiple targets fall in same bin, this logic is ambiguous, 
            # but usually they separate.
            
            best_target = None
            min_dist = float('inf')
            
            # Find which target generated this hit (roughly) to get the Angles
            # This simulates the radar beam direction at detection time.
            # Calculate Unambiguous Velocity Interval (Nyquist)
            # fd = 2*vr/lambda. fd wraps at PRF.
            # vr wraps at PRF * lambda / 2
            v_ambig = self.prf * lambda_w / 2.0
            
            for target in targets:
                tr, taz, tel, tvr = self._get_target_state(target, current_time)
                dr = abs(tr - r_meas)
                
                # Velocity difference with wrapping
                dv_raw = tvr - v_meas
                # Wrap dv into [-v_ambig/2, v_ambig/2]
                dv = abs(dv_raw - v_ambig * round(dv_raw / v_ambig))
                
                # Widen gate for check: bin conversion might introduce quantization error > delta
                # delta_r is bin width. Error is at most 0.5 bin usually, but let's be generous
                if dr < 2.0 * self.delta_r and dv < 2.0 * self.delta_v:
                    best_target = (taz, tel)
                    break
            
            # if best_target is None and np.random.rand() < 0.05:
            #      print(f"DEBUG: Missed Match R={r_meas:.1f} V={v_meas:.1f}, First Target R={targets[0][-1].state_vector[0]**2...}")
            #########################################################
            ### TEMP: REMOVE ERROR FOR NOW
            if best_target:
                az_meas = best_target[0] #+ np.random.normal(0, np.radians(1.0)) # 1 deg error
                el_meas = best_target[1] #+ np.random.normal(0, np.radians(1.0))
            else:
                # False alarm? Random angles?
                az_meas = np.random.uniform(-np.pi, np.pi)
                el_meas = np.random.uniform(-np.pi/4, np.pi/4)

            # Measurement vector: [R, Az, El, V_meas] (matching our expected measurement model)
            # Typically StoneSoup uses StateVectors.
            # Let's assume a 4D measurement space.
            vector = StateVector([r_meas, az_meas, el_meas, v_meas])
            
            det = Detection(vector, timestamp=current_time)
            det.metadata['is_target'] = (best_target is not None)
            detections.append(det)
            
        return detections, rd_map
