import gymnasium as gym
from gymnasium import spaces
import numpy as np

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Lidar, Electrics

class BeamNGSlopeShieldEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, bng_home=r"C:\BeamNG\BeamNG.tech.v0.39.4.0", render_mode=None):
        super(BeamNGSlopeShieldEnv, self).__init__()

        self.bng_home = bng_home
        self.render_mode = render_mode
        
        # 1. Action Space: 1D Longitudinal Control [-1.0 s.d. 1.0]
        # Nilai positif (>0) = Pedal Gas (Throttle) [0.0 s.d. 1.0]
        # Nilai negatif (<0) = Pedal Rem (Brake) [0.0 s.d. 1.0]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 2. Observation Space: [Kecepatan (m/s), Jarak LiDAR (m), Sudut Pitch (rad)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi/2], dtype=np.float32),
            high=np.array([50.0, 150.0, np.pi/2], dtype=np.float32),
            dtype=np.float32
        )

        # 3. Parameter Fisik SA-TTC Shield
        self.g = 9.81
        self.a_max = 7.5         # Kapasitas deselerasi rem nominal (m/s^2)
        self.tau_react = 0.2     # Waktu latensi aktuasi (detik)
        self.d_margin = 3.5      # Buffer jarak aman berhenti total (meter)
        self.lane_half_width = 1.7
        self.shield_latched = False

        # 4. Inisialisasi Simulator & Kendaraan
        print("Membuka instance BeamNG.tech...")
        self.bng = BeamNGpy('localhost', 64256, home=self.bng_home)
        self.bng.open(launch=True)

        self.ego_start_pos = (489.20, 178.71, 132.47)
        self.ego_start_rot = (0, 0, 0.7071, 0.7071)
        self.step_count = 0
        self.max_steps = 250     # Langkah maksimum dinaikkan agar cukup untuk lintasan 120m

        self.scenario = Scenario('automation_test_track', 'drl_slope_shield')
        
        self.ego = Vehicle('ego_vehicle', model='etk800')
        self.scenario.add_vehicle(self.ego, pos=self.ego_start_pos, rot_quat=self.ego_start_rot)
        
        # Mobil rintangan diatur sejauh 120 meter di depan lajur
        obs_pos = (self.ego_start_pos[0] - 120.0, self.ego_start_pos[1], self.ego_start_pos[2])
        self.obstacle = Vehicle('obstacle_box', model='etkc')
        self.scenario.add_vehicle(self.obstacle, pos=obs_pos, rot_quat=self.ego_start_rot)

        self.scenario.make(self.bng)
        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()

        self.ego.set_shift_mode('arcade')

        self.electrics = Electrics()
        self.ego.attach_sensor('electrics', self.electrics)

        # LiDAR Bemper Depan (Rendering laser visual dimatikan untuk efisiensi komputasi)
        self.lidar = Lidar(
            name='front_lidar',
            bng=self.bng,
            vehicle=self.ego,
            requested_update_time=0.05,
            is_using_shared_memory=False,
            pos=(0.0, -2.3, 0.6),
            dir=(0.0, -1.0, 0.0),
            up=(0.0, 0.0, 1.0),
            vertical_resolution=16,
            horizontal_angle=40,
            vertical_angle=12,
            max_distance=150.0,
            is_visualised=False
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.shield_latched = False

        self.bng.scenario.restart()
        self.ego.set_shift_mode('arcade')

        self.ego.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
        self.bng.control.step(10)
        
        obs = self._get_obs()
        info = {"shield_triggered": False}
        return obs, info

    def _get_obs(self):
        self.ego.sensors.poll()
        vel = np.array(self.ego.state.get('vel', (0, 0, 0)))
        speed_ms = float(np.linalg.norm(vel))
        
        dir_vec = np.array(self.ego.state.get('dir', (1, 0, 0)))
        up_vec = np.array(self.ego.state.get('up', (0, 0, 1)))
        
        # Vektor basis 3D body kendaraan (Chassis Frame ISO 8855)
        norm_dir = np.linalg.norm(dir_vec)
        fwd_3d = dir_vec / norm_dir if norm_dir > 0 else np.array([1.0, 0.0, 0.0])
        norm_up = np.linalg.norm(up_vec)
        up_3d = up_vec / norm_up if norm_up > 0 else np.array([0.0, 0.0, 1.0])
        left_3d = np.cross(up_3d, fwd_3d)
        norm_left = np.linalg.norm(left_3d)
        left_3d = left_3d / norm_left if norm_left > 0 else np.array([0.0, 1.0, 0.0])

        pitch_rad = float(np.arcsin(np.clip(fwd_3d[2], -1.0, 1.0)))

        ego_pos = np.array(self.ego.state.get('pos', (0, 0, 0)))
        bumper_pos = ego_pos + (fwd_3d * 2.3)
        
        data = self.lidar.poll()
        points = data.get('pointCloud', None)
        
        min_dist = 150.0
        if points is not None and len(points) > 0:
            rel_points = points - bumper_pos
            dist_fwd = np.dot(rel_points, fwd_3d)
            dist_lat = np.dot(rel_points, left_3d)
            dist_vert = np.dot(rel_points, up_3d)

            # Koridor lajur lateral [-lane_half_width, +lane_half_width]
            lane_mask = (dist_fwd > 0.5) & (dist_fwd < 150.0) & (np.abs(dist_lat) <= self.lane_half_width)
            fwd_lane = dist_fwd[lane_mask]
            vert_lane = dist_vert[lane_mask]

            if len(fwd_lane) > 0:
                # Segmentasi objek vertikal (menolak pantulan aspal/puncak tanjakan yang tipis)
                # Objek fisik (mobil/halangan) memiliki ketebalan vertikal >= 0.25m
                sort_idx = np.argsort(fwd_lane)
                fwd_s = fwd_lane[sort_idx]
                vert_s = vert_lane[sort_idx]

                # Binning dinamis per 2 meter
                bins = np.arange(1.0, 150.0, 2.0)
                for i in range(len(bins) - 1):
                    in_b = (fwd_s >= bins[i]) & (fwd_s < bins[i+1])
                    if np.sum(in_b) >= 4:
                        if (np.max(vert_s[in_b]) - np.min(vert_s[in_b])) >= 0.25:
                            min_dist = float(np.min(fwd_s[in_b]))
                            break

        return np.array([speed_ms, min_dist, pitch_rad], dtype=np.float32)

    def calculate_d_safe(self, v_ms, pitch_rad):
        effective_decel = self.a_max + (self.g * np.sin(pitch_rad))
        effective_decel = max(effective_decel, 1.0)
        d_safe = (v_ms**2) / (2.0 * effective_decel) + (v_ms * self.tau_react) + self.d_margin
        return d_safe

    def step(self, action):
        self.step_count += 1
        
        # Ekstraksi kontrol 1D longitudinal
        raw_throttle_brake = float(action[0])
        fixed_steering = 0.0  # Setir dikunci lurus agar mobil tidak zigzag

        if raw_throttle_brake >= 0:
            cmd_throttle = raw_throttle_brake
            cmd_brake = 0.0
        else:
            cmd_throttle = 0.0
            cmd_brake = abs(raw_throttle_brake)

        current_obs = self._get_obs()
        speed_ms, dist_lidar, pitch_rad = current_obs[0], current_obs[1], current_obs[2]
        
        # Evaluasi Slope-Aware Safety Shield (SA-TTC)
        d_safe = self.calculate_d_safe(speed_ms, pitch_rad)
        shield_intervened = False
        cmd_parkingbrake = 0.0

        if dist_lidar <= d_safe:
            self.shield_latched = True

        if self.shield_latched:
            shield_intervened = True
            cmd_throttle = 0.0
            if speed_ms > 0.8:
                cmd_brake = 1.0
                cmd_parkingbrake = 0.0
            else:
                cmd_brake = 0.0
                cmd_parkingbrake = 1.0
                if dist_lidar > (d_safe + 2.0):
                    self.shield_latched = False

        # Terapkan kontrol ke simulator
        self.ego.control(
            throttle=cmd_throttle,
            brake=cmd_brake,
            steering=fixed_steering,
            parkingbrake=cmd_parkingbrake
        )
        self.bng.control.step(15)

        next_obs = self._get_obs()
        new_speed, new_dist, _ = next_obs[0], next_obs[1], next_obs[2]

        # Fungsi Reward: Adaptif antara Cruising Zone dan Deceleration Zone
        target_speed_ms = 13.89
        if new_dist > (d_safe + 15.0):
            # Cruising zone: dorong untuk melaju mendekati target speed
            reward_speed = -abs(new_speed - target_speed_ms) / target_speed_ms
        else:
            # Deceleration zone: dorong deselerasi halus menuju nol saat mendekati d_margin
            target_approach_speed = max(0.0, target_speed_ms * ((new_dist - self.d_margin) / 25.0))
            reward_speed = -abs(new_speed - target_approach_speed) / target_speed_ms

        reward_shield_penalty = -5.0 if shield_intervened else 0.0
        reward = reward_speed + reward_shield_penalty

        terminated = False
        truncated = False
        is_collision = False
        is_safe_stop = False

        # Tabrakan keras
        if new_dist < 0.8 and new_speed > 0.8:
            reward -= 50.0
            terminated = True
            is_collision = True
        
        # Berhenti sempurna di depan rintangan (kecepatan < 0.2 m/s setelah fase akselerasi awal)
        if self.step_count > 5 and new_speed < 0.2:
            if new_dist <= 15.0:
                # Bonus ideal: jika berhenti di sekitar buffer d_margin (2.0m - 6.0m)
                if 2.0 <= new_dist <= 6.0:
                    reward += 25.0 if not self.shield_latched else 15.0
                else:
                    reward += 15.0 if not self.shield_latched else 8.0
                terminated = True
                is_safe_stop = True
            elif self.shield_latched:
                # Berhenti darurat aman terkunci oleh shield
                reward += 5.0
                terminated = True
                is_safe_stop = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "shield_triggered": shield_intervened,
            "d_safe": d_safe,
            "dist_actual": new_dist,
            "collision": is_collision,
            "safe_stop": is_safe_stop
        }

        return next_obs, reward, terminated, truncated, info

    def close(self):
        if self.bng:
            self.bng.close()