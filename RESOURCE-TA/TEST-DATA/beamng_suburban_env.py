import os
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Lidar, Electrics, Damage, Ultrasonic
from waypoint_manager import WaypointManager

class IntegratedSafetyShield:
    """
    Modul Pengawas Keselamatan Terpadu (Hierarchical Safety Shield):
    
    1. Polisi Laning (Control Barrier Enforcer / CBE) - Paramount Constraint:
       - Memantau batas koridor spasial jalan (d_curb) dan Lateral Position Variation (LPV).
       - Menjamin batas keselamatan mutlak (Safety Invariance h(x) >= 0) pada batas fisik roda & bodi.
       - Memotong paksa sudut kemudi (steer veto & barrier projection) jika manuver usulan mendekati
         bibir trotoar atau marka pembatas lajur luar.
         
    2. Polisi Stabilitas (Control Lyapunov Stabilizer / CLS) - Secondary Constraint:
       - Mencegah fenomena chattering / bang-bang control akibat intervensi setir agresif.
       - Slew-Rate Limiter: Membatasi laju putaran setir per langkah waktu
         (|delta_t - delta_{t-1}| <= Delta_delta_max) agar suspensi tidak tertekan mendadak.
       - Dynamic Cornering Speed Governor: Membatasi throttle dan deselerasi halus saat
         setir berbelok tajam (|delta| > 0.35 atau |curv| > 0.05, v > 12 km/jam)
         agar traksi lateral ban tidak terlampaui dan sasis tidak mengalami body roll berlebih.
       - Curb Deceleration Shield: Membuang energi kinetik kendaraan ketika
         mendekati trotoar pada kecepatan tinggi.
    """
    def __init__(
        self,
        lane_half_width=1.70,
        d_barrier_margin=0.22,
        max_steer_rate=0.35,
        governor_steer_threshold=0.35,
        governor_speed_kmh=12.0
    ):
        self.lane_half_width = lane_half_width
        self.d_barrier_margin = d_barrier_margin
        self.max_steer_rate = max_steer_rate
        self.governor_steer_threshold = governor_steer_threshold
        self.governor_speed_kmh = governor_speed_kmh
        self.last_steer = 0.0

    def reset(self):
        self.last_steer = 0.0

    def enforce(
        self,
        raw_steering: float,
        raw_throttle: float,
        raw_brake: float,
        d_curb_phys_right: float,
        d_curb_phys_left: float,
        d_curb_eff: float,
        speed_ms: float,
        e_yaw: float = 0.0,
        curv: float = 0.0,
        v_target_ms: float = 20.0
    ):
        cbe_intervened = False
        cls_intervened = False
        speed_kmh = speed_ms * 3.6

        # =================================================================
        # POLISI 1: LANING SHIELD (Control Barrier Enforcer - CBE)
        # Prioritas Utama (Paramount Constraint: Safety Invariance h(x) >= 0)
        # =================================================================
        cbe_steer = raw_steering

        # Hitung proyeksi laju pendekatan lateral roda ke batas jalan (Approach Vector):
        # Laju pendekatan ke trotoar kanan: dipengaruhi arah hadap sasis (e_yaw) dan sudut kemudi (cbe_steer)
        # cbe_steer > 0 = belok kanan, e_yaw > 0 = hidung menghadap kanan
        approach_rate_right = e_yaw + 0.40 * cbe_steer
        approach_rate_left  = -e_yaw - 0.40 * cbe_steer

        # Skalakan koreksi CBE berdasarkan kecepatan (Anti-Ping-Pong di Kecepatan Tinggi):
        # Di atas 25 km/jam, batasi intervensi maksimum (max 0.12) agar mobil tidak membanting liar melintasi marka tengah.
        # Di bawah 25 km/jam di tikungan, berikan otoritas penuh (0.32) untuk mencegah tabrakan trotoar.
        if speed_kmh > 25.0:
            cbe_gain = 0.12
            emergency_nudge = 0.05
        else:
            cbe_gain = 0.32
            emergency_nudge = 0.15

        # PILAR 4B: CBE LANING KORIDOR (Clamping Tangensial, Bukan Membalik Arah Kemudi)
        # Sisi Kanan: Dekat trotoar kanan / batas luar jalan
        if d_curb_phys_right < self.d_barrier_margin:
            # HANYA lakukan veto jika kendaraan sedang mengarah/membelok mendekati trotoar kanan!
            # Jika kendaraan sudah miring ke kiri (e_yaw < -0.10 rad) akibat tail-swing,
            # kemudi kanan justru WAJIB diizinkan agar hidung mobil dapat kembali sejajar lajur jalan.
            if approach_rate_right > 0.0:
                violation_r = self.d_barrier_margin - d_curb_phys_right
                # Clamping tangensial: batasi sudut belok kanan (steer > 0) agar tidak memotong trotoar,
                # tetapi JANGAN klem terlalu sempit yang memotong radius putar mobil di tikungan tajam!
                if cbe_steer > 0.0:
                    max_steer_cap = 1.0 if (speed_kmh < 15.0 or abs(curv) > 0.02 or abs(e_yaw) > 0.12) else 0.70
                    max_safe_steer_r = float(np.clip(max_steer_cap * (1.0 - violation_r / self.d_barrier_margin), 0.0, max_steer_cap))
                    if cbe_steer > max_safe_steer_r:
                        cbe_steer = max_safe_steer_r
                        cbe_intervened = True
            # Torsi dorong darurat dengan Deadband & Tapering Halus (Anti-Ping-Pong):
            # Pada kecepatan > 15 km/jam, redam total dorongan sentak agar tidak overshoot ke marka kiri.
            # Pada kecepatan rendah, dorong lembut HANYA jika jarak < 5 cm dan moncong mobil masih mengarah ke dinding.
            if speed_kmh < 15.0 and d_curb_phys_right < 0.05 and approach_rate_right > 0.12:
                nudge_r = float(np.clip(emergency_nudge * ((0.05 - d_curb_phys_right) / 0.05), 0.0, 0.08))
                cbe_steer -= nudge_r
                cbe_intervened = True

        # Sisi Kiri: Dekat batas lajur kiri / garis marka
        elif d_curb_phys_left < self.d_barrier_margin:
            # HANYA lakukan veto jika kendaraan sedang mengarah/membelok mendekati marka kiri!
            if approach_rate_left > 0.0:
                violation_l = self.d_barrier_margin - d_curb_phys_left
                # Clamping tangensial: batasi sudut belok kiri (steer < 0) agar tidak melintasi marka tengah,
                # tetapi JANGAN klem terlalu sempit yang memotong radius putar mobil!
                if cbe_steer < 0.0:
                    min_steer_cap = 1.0 if (speed_kmh < 15.0 or abs(curv) > 0.02 or abs(e_yaw) > 0.12) else 0.70
                    min_safe_steer_l = float(np.clip(-min_steer_cap * (1.0 - violation_l / self.d_barrier_margin), -min_steer_cap, 0.0))
                    if cbe_steer < min_safe_steer_l:
                        cbe_steer = min_safe_steer_l
                        cbe_intervened = True
            # Torsi dorong darurat dengan Deadband & Tapering Halus (Anti-Ping-Pong):
            # Pada kecepatan > 15 km/jam, redam total dorongan sentak agar tidak overshoot ke trotoar kanan.
            # Pada kecepatan rendah, dorong lembut HANYA jika jarak < 5 cm dan moncong mobil masih mengarah ke marka.
            if speed_kmh < 15.0 and d_curb_phys_left < 0.05 and approach_rate_left > 0.12:
                nudge_l = float(np.clip(emergency_nudge * ((0.05 - d_curb_phys_left) / 0.05), 0.0, 0.08))
                cbe_steer += nudge_l
                cbe_intervened = True

        cbe_steer = float(np.clip(cbe_steer, -1.0, 1.0))

        # =================================================================
        # POLISI 2: STABILITY SHIELD (Control Lyapunov Stabilizer - CLS)
        # Prioritas Sekunder (Secondary Constraint: Dynamic Smoothness & Grip)
        # =================================================================
        # PILAR 3: ADAPTIVE SLEW-RATE CLS (Anti-Phase Lag & Anti-Ping-Pong / Anti-Fishtailing)
        # Slew rate cepat (0.40) HANYA diizinkan pada kecepatan rendah dan tikungan tajam yang nyata.
        # Pada jalan lurus / kurva sedang, batasi pembalikan arah setir agar tidak timbul osilasi bola ping-pong!
        is_true_sharp_corner = (abs(curv) > 0.025 or abs(e_yaw) > 0.20) and speed_kmh < 22.0

        if is_true_sharp_corner:
            effective_max_steer_rate = 0.40  # Otoritas putaran setir cepat untuk tikungan 90 derajat
        elif speed_kmh > 35.0:
            effective_max_steer_rate = 0.12  # Redaman solid di jalan raya cepat bebas goyang
        else:
            effective_max_steer_rate = 0.20  # Kecepatan jelajah menengah stabil, redam osilasi bolak-balik

        delta_steer = cbe_steer - self.last_steer
        clipped_delta = float(np.clip(delta_steer, -effective_max_steer_rate, effective_max_steer_rate))
        if abs(delta_steer) > effective_max_steer_rate:
            cls_intervened = True
        final_steering = float(np.clip(self.last_steer + clipped_delta, -1.0, 1.0))
        self.last_steer = final_steering

        # B. Cornering Speed Governor: Batasi overspeed saat setir terbelok tajam
        final_throttle = raw_throttle
        final_brake = raw_brake

        # Governor hanya aktif jika kecepatan kendaraan MELEBIHI batas aman profil kinematis (v_target + 3 km/jam)
        v_target_kmh = v_target_ms * 3.6
        allowed_corner_spd = max(self.governor_speed_kmh, v_target_kmh + 3.0)

        if (abs(final_steering) > self.governor_steer_threshold or abs(curv) > 0.05) and speed_kmh > allowed_corner_spd:
            cls_intervened = True
            final_throttle = 0.0
            excess_speed_kmh = speed_kmh - allowed_corner_spd
            if excess_speed_kmh > 4.0 and speed_ms > 1.5:
                gov_brake = float(np.clip(excess_speed_kmh / 15.0, 0.15, 0.50))
                final_brake = max(final_brake, gov_brake)

        # C. Curb Deceleration Protection (Proteksi Energi Kinetik Bibir Trotoar)
        # Hanya aktif jika mendekati trotoar pada kecepatan tinggi (> 2.5 m/s ~ 9 km/jam)
        if d_curb_eff < 0.28 and speed_ms > 2.5:
            final_throttle = min(final_throttle, 0.10)
            curb_brake = float(np.clip((0.28 - d_curb_eff) / 0.28, 0.15, 0.45))
            final_brake = max(final_brake, curb_brake)

        # D. Mekanisme Escape Crawl CBE (Pencegah Deadlock VEHICLE_STUCK di Bibir Trotoar):
        # Saat mobil bergerak sangat lambat (< 1.5 km/jam) dan menempel bibir trotoar:
        # 1. Lepas rem 100% penuh (final_brake = 0.0) agar roda bebas berputar maju.
        # 2. Suntikkan torsi jelajah merayap minimum (unstick crawl throttle >= 0.25) untuk
        #    mengatasi hambatan gesek statis sasis dan membebaskan mobil dari trotoar.
        # 3. KEMUDI: Bebaskan sudut setir penuh (hingga +/- 1.0) agar mobil mampu melipat
        #    tikungan tajam dan melepaskan diri dari trotoar tanpa dipotong radius putarnya!
        if speed_kmh < 1.5 and (d_curb_phys_right <= 0.08 or d_curb_phys_left <= 0.08):
            final_brake = 0.0
            final_throttle = max(final_throttle, 0.25)
            final_steering = float(np.clip(final_steering, -1.0, 1.0))
            self.last_steer = final_steering

        return final_steering, final_throttle, final_brake, cbe_intervened, cls_intervened


class BeamNGSuburbanEnv(gym.Env):
    """
    Lingkungan Gymnasium Otonom Terintegrasi (Hierarchical Autonomous System):
      - Kendali Lateral: Stanley Controller berbasis Waypoint (menjaga mobil di lajur)
      - Kendali Longitudinal: DRL (PPO) + Slope-Aware Safety Shield (SA-TTC)
      - Skenario: Lintasan berbelok Sub-Urban / Mountain dengan rintangan kendaraan (Statis atau Bergerak)
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        bng_home=r"C:\BeamNG\BeamNG.tech.v0.39.4.0",
        route_type='suburban',
        start_offset_m=45.0,
        obstacle_distance_m=120.0,
        is_moving_obstacle=False,
        has_obstacle=True,
        target_cruise_kmh=80.0,
        render_mode=None,
        use_yolo=True,
        use_progressive_spawn=False,
        progressive_spawn_checkpoints=None
    ):
        super(BeamNGSuburbanEnv, self).__init__()

        self.bng_home = bng_home
        self.route_type = route_type
        self.start_offset_m = start_offset_m
        self.obstacle_distance_m = obstacle_distance_m
        self.is_moving_obstacle = is_moving_obstacle
        self.has_obstacle = has_obstacle
        self.target_cruise_kmh = target_cruise_kmh
        self.render_mode = render_mode
        self.use_yolo = use_yolo
        self.use_progressive_spawn = use_progressive_spawn
        self.progressive_spawn_checkpoints = progressive_spawn_checkpoints or [48.0, 180.0, 350.0, 700.0, 1200.0]

        # 1. Konfigurasi Rute dan Peta
        if self.route_type == 'suburban':
            self.map_name = 'east_coast_usa'
            self.csv_name = 'waypoints_suburban_town.csv'
            self.k_e = 1.20
            self.k_soft = 0.80
        elif self.route_type == 'mountain_castle':
            self.map_name = 'italy'
            self.csv_name = 'waypoints_italy_castle.csv'
            self.k_e = 0.70
            self.k_soft = 1.50
        else:
            self.map_name = 'italy'
            self.csv_name = 'waypoints_italy_mountain.csv'
            self.k_e = 0.65
            self.k_soft = 1.50

        # Muat dataset waypoints dan potong offset awal
        csv_path = os.path.join(os.path.dirname(__file__), self.csv_name)
        self.df_raw = pd.read_csv(csv_path)
        self.df_wp = self.df_raw[self.df_raw['distance_m'] >= self.start_offset_m].reset_index(drop=True)
        self.waypoints = self.df_wp[['x', 'y', 'z']].values

        # Inisialisasi WaypointManager untuk kendali lateral otomatis (Speed-Adaptive Stanley + Center of Gravity Tracking)
        # serta Kinematic Speed Profiler untuk jelajah halus hingga target_cruise_kmh (80 km/jam)
        # use_dynamic_reference=True: Mengaktifkan transisi CoM (jalan lurus) <-> CoRA + Cornering Expansion (tikungan tajam)
        self.wm = WaypointManager(
            waypoints=self.waypoints,
            k_e=self.k_e,
            k_soft=self.k_soft,
            front_axle_dist=0.0,
            use_dynamic_reference=True,
            lateral_offset_m=0.0,
            target_cruise_kmh=self.target_cruise_kmh
        )

        # 2. Action Space: 1D Longitudinal Control [-1.0 s.d. 1.0]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 3. Observation Space: [Kecepatan (m/s), Jarak Rintangan (m), Sudut Pitch (rad)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi/2], dtype=np.float32),
            high=np.array([50.0, 150.0, np.pi/2], dtype=np.float32),
            dtype=np.float32
        )

        # 4. Parameter Fisik SA-TTC Shield & Pengawas Keselamatan Terpadu (CBE + CLS)
        self.g = 9.81
        self.a_max = 7.5         # Kapasitas deselerasi rem nominal (m/s^2)
        self.tau_react = 0.2     # Waktu latensi aktuasi (detik)
        self.d_margin = 3.5      # Buffer jarak aman berhenti total (meter)
        self.lane_half_width = 1.7
        self.shield_latched = False

        self.safety_shield = IntegratedSafetyShield(
            lane_half_width=self.lane_half_width,
            d_barrier_margin=0.22,
            max_steer_rate=0.35,
            governor_steer_threshold=0.35,
            governor_speed_kmh=12.0
        )

        self.step_count = 0
        self.max_steps = 250
        self.last_v_target = 6.5 / 3.6
        self.last_steer_action = 0.0

        # 5. Inisialisasi Simulator & Kendaraan
        print(f"Membuka instance BeamNG.tech pada peta '{self.map_name}' ({self.route_type})...")
        self.bng = BeamNGpy('localhost', 64256, home=self.bng_home)
        self.bng.open(launch=True)

        self.ego_start_pos, self.ego_start_rot = WaypointManager.calculate_start_pose(self.df_wp)

        # Hitung posisi spawn mobil rintangan di depan pada lajur jalan
        cum_dists = self.df_wp['distance_m'].values - self.df_wp['distance_m'].iloc[0]
        obs_idx = int(np.searchsorted(cum_dists, self.obstacle_distance_m))
        obs_idx = min(obs_idx, len(self.df_wp) - 2)
        
        self.obs_start_pos = (
            float(self.df_wp['x'].iloc[obs_idx]),
            float(self.df_wp['y'].iloc[obs_idx]),
            float(self.df_wp['z'].iloc[obs_idx]) + 0.15
        )
        dx_obs = float(self.df_wp['x'].iloc[obs_idx+1] - self.df_wp['x'].iloc[obs_idx])
        dy_obs = float(self.df_wp['y'].iloc[obs_idx+1] - self.df_wp['y'].iloc[obs_idx])
        theta_obs = np.arctan2(-dx_obs, -dy_obs)
        self.obs_start_rot = (0.0, 0.0, float(np.sin(theta_obs / 2.0)), float(np.cos(theta_obs / 2.0)))

        self.scenario = Scenario(self.map_name, 'drl_suburban_shield')
        self.ego = Vehicle('ego_vehicle', model='etk800')
        self.scenario.add_vehicle(self.ego, pos=self.ego_start_pos, rot_quat=self.ego_start_rot)

        if self.has_obstacle:
            self.obstacle = Vehicle('obstacle_box', model='etkc')
            self.scenario.add_vehicle(self.obstacle, pos=self.obs_start_pos, rot_quat=self.obs_start_rot)
        else:
            self.obstacle = None

        self.scenario.make(self.bng)
        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()

        self.ego.set_shift_mode('arcade')

        # Konfigurasi transmisi obstacle (jika ada)
        if self.has_obstacle and self.obstacle is not None:
            if self.is_moving_obstacle:
                self.obstacle.set_shift_mode('arcade')
                try:
                    self.obstacle.ai.set_mode('span')
                    self.obstacle.ai.drive_in_lane(True)
                    self.obstacle.ai.set_speed(5.5)  # 20 km/jam maju di lajur jalan
                except Exception as e:
                    print(f"Warning: Obstacle AI mode: {e}")
            else:
                try:
                    self.obstacle.set_shift_mode('realistic_manual')
                except Exception:
                    pass
                self.obstacle.control(gear=0, throttle=0.0, brake=1.0, parkingbrake=1.0)

        self.electrics = Electrics()
        self.ego.attach_sensor('electrics', self.electrics)
        self.damage = Damage()
        self.ego.attach_sensor('damage', self.damage)

        # LiDAR Bemper Depan
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
            horizontal_angle=120,
            vertical_angle=20,
            max_distance=150.0,
            is_visualised=False
        )

        # Sensor Ultrasonic Sisi Kanan (Dimatikan: Digantikan sepenuhnya oleh Kamera YOLOv8-seg)
        self.us_front_right = None
        self.us_rear_right = None

        # Sensor Kamera Kaca Depan & Modul Ekstraktor Spasial YOLOv8-seg (Frozen Mode)
        self.camera = None
        self.yolo_extractor = None
        self.last_d_curb_yolo = 4.0
        self.last_ey_yolo = 0.0
        self.last_epsi_yolo = 0.0
        self.last_yolo_hud_frame = None

        if self.use_yolo:
            try:
                from yolo_spatial_extractor import YOLOSpatialExtractor
                from beamngpy.sensors import Camera
                print("-> Mengaktifkan Kamera Bemper Depan & YOLOv8-seg (Frozen Inference)...")
                self.camera = Camera(
                    name='front_yolo_cam',
                    bng=self.bng,
                    vehicle=self.ego,
                    requested_update_time=0.1,
                    pos=(0.0, -2.15, 0.85),
                    dir=(0.0, -1.0, -0.22),
                    up=(0.0, 0.0, 1.0),
                    resolution=(640, 480),
                    field_of_view_y=70,
                    near_far_planes=(0.1, 120.0),
                    is_using_shared_memory=False,
                    is_render_colours=True,
                    is_render_annotations=False,
                    is_render_depth=False,
                    is_visualised=False
                )
                self.yolo_extractor = YOLOSpatialExtractor()
                print("-> Modul Persepsi Visual YOLOv8-seg BERHASIL diaktifkan!")
            except Exception as e_cam:
                print(f"-> Info: Inisialisasi Kamera YOLO ({e_cam}) fallback ke mode LiDAR/Geometris.")
                self.camera = None
                self.yolo_extractor = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.shield_latched = False
        self.stalled_steps = 0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_steering = 0.0
        self.last_steer_action = 0.0
        self.last_v_target = 6.5 / 3.6
        self.safety_shield.reset()
        self.wm.last_closest_idx = 0
        self.wm.prev_steering = 0.0
        self.wm.prev_ego_yaw = None
        self.bng.scenario.restart()
        self.ego.set_shift_mode('arcade')

        # PROGRESSIVE MULTI-SECTOR SPAWN (Kurikulum Latihan Multi-Sektor)
        if getattr(self, 'use_progressive_spawn', False) and getattr(self, 'progressive_spawn_checkpoints', None):
            selected_offset = float(np.random.choice(self.progressive_spawn_checkpoints))
            self.start_offset_m = selected_offset
            self.df_wp = self.df_raw[self.df_raw['distance_m'] >= selected_offset].reset_index(drop=True)
            self.waypoints = self.df_wp[['x', 'y', 'z']].values
            self.wm.set_waypoints(self.waypoints, target_cruise_kmh=self.target_cruise_kmh)
            new_pos, new_rot = WaypointManager.calculate_start_pose(self.df_wp)
            self.ego.teleport(pos=new_pos, rot_quat=new_rot, reset=True)
            self.bng.control.step(5)
            print(f"  [SPAWN SECTOR] Mobil di-spawn di Sektor Offset {selected_offset:.1f}m (Sisa rute: {len(self.df_wp)} WP)")

        self.prev_distance_m = float(self.df_wp['distance_m'].iloc[0])

        if self.has_obstacle and self.obstacle is not None:
            if self.is_moving_obstacle:
                self.obstacle.set_shift_mode('arcade')
                try:
                    self.obstacle.ai.set_mode('span')
                    self.obstacle.ai.drive_in_lane(True)
                    self.obstacle.ai.set_speed(5.5)
                except Exception:
                    pass
            else:
                try:
                    self.obstacle.set_shift_mode('realistic_manual')
                except Exception:
                    pass
                self.obstacle.control(gear=0, throttle=0.0, brake=1.0, steering=0.0, parkingbrake=1.0)

        self.ego.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
        self.bng.control.step(10)

        obs = self._get_obs()
        info = {"shield_triggered": False}
        return obs, info

    def _get_obs(self):
        self.ego.sensors.poll()
        if self.has_obstacle and self.obstacle is not None:
            self.obstacle.sensors.poll()

        vel = np.array(self.ego.state.get('vel', (0, 0, 0)))
        speed_ms = float(np.linalg.norm(vel))

        dir_vec = np.array(self.ego.state.get('dir', (1, 0, 0)))
        up_vec = np.array(self.ego.state.get('up', (0, 0, 1)))

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

        # Hitung jarak rintangan di sepanjang lintasan jalan (Path-Distance) jika ada rintangan
        path_lead_dist = 150.0
        obs_dist_along_path = 9999.0
        ego_dist_along_path = 0.0

        if self.has_obstacle and self.obstacle is not None:
            obs_pos = np.array(self.obstacle.state.get('pos', (0, 0, 0)))
            ego_idx = self.wm.find_closest_index(ego_pos)
            obs_diffs = self.wm.xy - obs_pos[:2]
            obs_idx = int(np.argmin(np.sum(obs_diffs**2, axis=1)))
            
            ego_dist_along_path = float(self.df_wp['distance_m'].iloc[ego_idx])
            obs_dist_along_path = float(self.df_wp['distance_m'].iloc[obs_idx])
            path_lead_dist = max(0.0, obs_dist_along_path - ego_dist_along_path - 4.5)

        data = self.lidar.poll()
        points = data.get('pointCloud', None)

        # Sudut kemiringan melintang bodi (Roll Angle) dalam derajat
        # left_3d[2] adalah elevasi sumbu lateral mobil (tanda fisik jika salah satu roda naik trotoar)
        roll_rad = float(np.arcsin(np.clip(left_3d[2], -1.0, 1.0)))
        self.last_roll_deg = float(np.degrees(roll_rad))

        # Solusi A (Mata Proaktif): Deteksi Undakan Trotoar Kanan via LiDAR Height-Step
        # LiDAR di Z=0.6m. Aspal di dist_vert ≈ -0.60m. Undakan 5-15 cm di [-0.55, -0.40]m
        d_curb_lidar = 4.0

        lidar_min = 150.0
        if points is not None and len(points) > 0:
            rel_points = points - bumper_pos
            dist_fwd = np.dot(rel_points, fwd_3d)
            dist_lat = np.dot(rel_points, left_3d)
            dist_vert = np.dot(rel_points, up_3d)

            # LiDAR difokuskan murni untuk deteksi mobil/rintangan depan di dalam lajur jalan
            lane_mask = (dist_fwd > 0.2) & (dist_fwd < 150.0) & (np.abs(dist_lat) <= self.lane_half_width)
            fwd_lane = dist_fwd[lane_mask]
            vert_lane = dist_vert[lane_mask]

            if len(fwd_lane) > 0:
                sort_idx = np.argsort(fwd_lane)
                fwd_s = fwd_lane[sort_idx]
                vert_s = vert_lane[sort_idx]

                bins = np.arange(0.2, 150.0, 1.0)
                for i in range(len(bins) - 1):
                    in_b = (fwd_s >= bins[i]) & (fwd_s < bins[i+1])
                    if np.sum(in_b) >= 3:
                        if (np.max(vert_s[in_b]) - np.min(vert_s[in_b])) >= 0.18:
                            lidar_min = float(np.min(fwd_s[in_b]))
                            break

        self.last_d_curb_lidar = d_curb_lidar

        # 2. Polling Kamera Bemper & Ekstraksi Metrik Spasial YOLOv8-seg (Frozen Mode)
        if self.use_yolo and self.camera is not None and self.yolo_extractor is not None:
            try:
                cam_data = self.camera.poll()
                if cam_data and 'colour' in cam_data and cam_data['colour'] is not None:
                    img_np = np.array(cam_data['colour'].convert('RGB'))
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    metrics = self.yolo_extractor.extract_metrics(img_bgr)
                    self.last_d_curb_yolo = metrics.d_curb
                    self.last_ey_yolo = metrics.e_y
                    self.last_epsi_yolo = metrics.e_psi
                    self.last_yolo_hud_frame = self.yolo_extractor.get_annotated_hud()
                else:
                    if not getattr(self, '_cam_warn_logged', False):
                        print(f"  [PERINGATAN KAMERA] Polling kamera berhasil tetapi data colour belum tersedia.")
                        self._cam_warn_logged = True
            except Exception as e_cam_poll:
                if not getattr(self, '_cam_err_logged', False):
                    import traceback
                    print(f"  [ERROR KAMERA] Exception polling kamera bemper: {e_cam_poll}")
                    traceback.print_exc()
                    self._cam_err_logged = True

        # Tentukan jarak efektif ke objek depan (Multi-Modal: LiDAR + Path-Distance + Direct Euclidean)
        if self.has_obstacle and self.obstacle is not None:
            # 1. Jarak Euclidean bumper-ke-bumper aktual
            direct_dist = max(0.0, float(np.linalg.norm(obs_pos[:2] - bumper_pos[:2]) - 2.2))
            
            # 2. Periksa apakah rintangan berada di sektor depan kendaraan
            rel_to_obs = obs_pos[:2] - ego_pos[:2]
            fwd_dot = float(np.dot(fwd_3d[:2], rel_to_obs))
            
            candidates = [lidar_min]
            if obs_dist_along_path >= ego_dist_along_path:
                candidates.append(path_lead_dist)
            if fwd_dot > 0.0:  # Mobil rintangan berada di sektor depan kendaraan ego
                candidates.append(direct_dist)
            
            final_dist = float(min(candidates))
        else:
            # Mode Solo Cruising: Jalan kosong
            # Emergency Bumper Guard: Jika ada tiang/tembok/rintangan kritis < 2.5m tepat di depan bumper
            if lidar_min < 2.5:
                final_dist = lidar_min
            else:
                final_dist = 150.0

        return np.array([speed_ms, final_dist, pitch_rad], dtype=np.float32)

    def calculate_d_safe(self, v_ms, pitch_rad):
        effective_decel = self.a_max + (self.g * np.sin(pitch_rad))
        effective_decel = max(effective_decel, 1.0)
        d_safe = (v_ms**2) / (2.0 * effective_decel) + (v_ms * self.tau_react) + self.d_margin
        return d_safe

    def step(self, action):
        self.step_count += 1

        # 1. Kontrol Lateral Otomatis (Stanley Controller)
        ego_pos = np.array(self.ego.state.get('pos', (0, 0, 0)))
        dir_vec = np.array(self.ego.state.get('dir', (1, 0, 0)))
        vel_vec = np.array(self.ego.state.get('vel', (0, 0, 0)))
        speed_ms = float(np.linalg.norm(vel_vec))
        ego_yaw = float(np.arctan2(dir_vec[1], dir_vec[0]))

        cmd_steering, e_lat, e_yaw = self.wm.compute_steering(ego_pos, ego_yaw, speed_ms)

        # 2. Profil Kecepatan Kinematis & Horizon Deselerasi Dinamis (Dynamic Braking Lookahead)
        closest_idx = self.wm.find_closest_index(ego_pos)
        v_target = self.wm.get_dynamic_lookahead_target_speed(closest_idx, speed_ms)
        if self.wm.curvatures is not None and closest_idx < len(self.wm.curvatures):
            curv = float(self.wm.curvatures[closest_idx])
        else:
            curv = self.wm.get_curvature(closest_idx)

        # Antisipasi Tikungan Tajam Jarak Kinetis (Kinetic Braking Margin Safeguard):
        # Tambahkan jarak pengereman antisipatif proporsional terhadap kuadrat kecepatan (v^2 / 2a)
        # Memberikan jeda antisipatif lebih awal bagi rem untuk memangkas kecepatan sebelum masuk mulut belokan.
        # Batasi kecepatan target maksimum <= 11.0 km/jam saat mendeteksi tikungan di depan.
        braking_margin_dist = max(20.0, (speed_ms ** 2) / (2.0 * 2.0))
        if self.wm.detect_sharp_corner_ahead(closest_idx, lookahead_m=braking_margin_dist, kappa_threshold=0.035):
            v_target = min(v_target, 11.0 / 3.6)

        # PILAR 4C: SPAWN-ZONE PROTECTION (Langkah 0 s.d. 25)
        # Mobil di-spawn 12 meter sebelum belokan pertama (offset 48m).
        # Kunci kecepatan di bawah 8 km/jam sampai suspensi stabil dan kurva pertama terlewati.
        if self.step_count < 25:
            v_target = min(v_target, 8.0 / 3.6)

        # Syarat Akselerasi Lurus (Straight-Line Acceleration Gate):
        # Mencegah akselerasi prematur saat mobil baru keluar tikungan tapi sudut hadap (e_yaw)
        # atau posisi lajur (e_lat) masih belum stabil.
        # Gas kencang saat bodi miring memicu power oversteer & transfer beban squat yang merusak traksi depan.
        if abs(e_yaw) > 0.12 or abs(e_lat) > 0.40:
            if speed_ms < (25.0 / 3.6):
                # Kunci target kecepatan maksimal 11.0 km/jam (<= 12 km/jam) hingga sasis tegak di tengah lajur
                v_target = min(v_target, 11.0 / 3.6)
            else:
                # Pada kecepatan tinggi di jalan lurus, tahan laju pada kecepatan saat ini tanpa memicu panic brake
                v_target = min(v_target, speed_ms)

        # 3. Rate Limiter Target Kecepatan (Acceleration Slew-Rate Limiter on v_target):
        # Mencegah lonjakan mendadak v_target (misal melompat dari 18 km/jam ke 58 km/jam dalam 1 step).
        # Akselerasi target dibatasi maksimal +2.5 m/s^2 (~0.375 m/s atau ~1.35 km/jam per step).
        # Deselerasi (penurunan target saat mendekati belokan) TIDAK dibatasi agar pengereman tetap tepat waktu.
        max_v_accel_step = 2.5 * 0.15  # ~0.375 m/s (~1.35 km/h per step)
        if hasattr(self, 'last_v_target') and self.last_v_target is not None:
            if v_target > self.last_v_target:
                v_target = min(v_target, self.last_v_target + max_v_accel_step)
        self.last_v_target = v_target

        # 4. Evaluasi Dual Safety Shield (Slope-Aware SA-TTC + Progressive Handling Shield)
        current_obs = self._get_obs()
        dist_lidar = current_obs[1]
        pitch_rad = current_obs[2]
        d_safe = self.calculate_d_safe(speed_ms, pitch_rad)

        # --- B. PHYSICAL CORRIDOR ERL SHIELDING (Fail-Safe Fallback Independen dari Visual) ---
        # Mengatasi masalah false-positive drivable area saat trotoar memiliki warna/tekstur mirip aspal:
        # Perhitungan jarak ke batas trotoar didasarkan pada selubung geometris fisik kendaraan (4 roda & bumper)
        # terhadap koridor jalan (lane half-width = 1.70m). Jika modul visi tertipu ilusi optik warna abu-abu,
        # koridor fisik ini tetap 100% menjamin roda dan bodi tidak memotong maupun memanjat trotoar.
        bbox = getattr(self.wm, 'last_bbox_errors', {})
        corridor_half_w = float(self.lane_half_width)  # 1.70m dari garis tengah ke bibir trotoar

        incursion_right = max(
            bbox.get("front_right", e_lat + 0.93),
            bbox.get("rear_right", e_lat + 0.93),
            bbox.get("bumper_rear_right", e_lat + 0.93)
        )
        incursion_left = max(
            -bbox.get("front_left", e_lat - 0.93),
            -bbox.get("rear_left", e_lat - 0.93),
            -bbox.get("bumper_rear_left", e_lat - 0.93)
        )

        raw_clearance_r = float(corridor_half_w - incursion_right)
        raw_clearance_l = float(corridor_half_w - incursion_left)
        raw_curb_clearance = min(raw_clearance_r, raw_clearance_l)

        d_curb_phys_right = max(0.0, raw_clearance_r)
        d_curb_phys_left  = max(0.0, raw_clearance_l)
        d_corridor_phys   = min(d_curb_phys_right, d_curb_phys_left)

        # Fusi Multi-Modal Fail-Safe:
        # Jika YOLO mengalami ilusi optik (merasa aman d_curb_visual = 4.0m),
        # raw_curb_clearance akan mengambil alih sebagai batas fisik absolut!
        # Nilai negatif menandakan roda menembus batas trotoar (Ground-Truth Penetration)
        d_curb_visual = getattr(self, 'last_d_curb_yolo', 4.0)
        d_curb_eff = min(d_curb_visual, raw_curb_clearance)

        raw_steering = cmd_steering
        shield_intervened = False
        cbe_intervened = False
        cls_intervened = False
        cmd_parkingbrake = 0.0

        # Galat kecepatan terhadap target kinematis (selalu tersedia di seluruh cabang shield)
        speed_error = speed_ms - v_target

        # A. Shield Darurat Rintangan Depan (Emergency Collision Avoidance)
        if self.has_obstacle and dist_lidar <= d_safe:
            self.shield_latched = True

        if self.shield_latched:
            shield_intervened = True
            cmd_throttle = 0.0
            cmd_brake = 1.0  # Tahan rem hidrolik 100% penuh tanpa kompromi (mencegah creeping)
            cmd_parkingbrake = 1.0 if speed_ms < 0.5 else 0.0
            if dist_lidar > (d_safe + 3.0):
                self.shield_latched = False
            self.prev_throttle = 0.0
            self.prev_brake = 1.0
            self.safety_shield.last_steer = cmd_steering
        else:
            # B. Kendali Longitudinal Progresif (DRL PPO + Kinematic Speed Tracking + Slew-Rate Limiter)
            raw_action = float(action[0])

            if speed_error > 0.3:
                # Mobil melebihi target kinematis (persiapan memasuki tikungan)
                shield_intervened = True
                target_throttle = 0.0
                # Deselerasi responsif proporsional (Anti-Overspeed Corner Entry)
                target_brake = float(np.clip(speed_error / 2.0, 0.15, 0.85))
            elif abs(speed_error) <= 0.3:
                # Kecepatan berada di sekitar target profil (jelajah stabil ~80 km/jam di jalan lurus)
                target_brake = 0.0
                # Modulasi gas proporsional untuk menahan laju kendaraan tanpa sentakan
                target_throttle = float(np.clip(0.18 - 0.18 * (speed_error / 0.3), 0.05, 0.30))
            else:
                # Kecepatan di bawah target profil (akselerasi menuju target kecepatan)
                target_brake = 0.0
                speed_cap = 0.30 if v_target < 4.5 else 1.0
                if abs(e_lat) < 0.35:
                    throttle_cap = speed_cap
                else:
                    throttle_cap = float(np.clip(speed_cap - (abs(e_lat) - 0.35) * 1.5, 0.20, speed_cap))
                # Jamin torsi mesin minimum pada kecepatan merayap agar mobil tidak mogok melawan gesekan belok penuh
                if speed_ms < 2.0:
                    throttle_cap = max(throttle_cap, 0.35)
                target_throttle = float(np.clip(min(max(raw_action, 0.20), throttle_cap), 0.0, 1.0))

            # Actuator Slew-Rate Limiter: Membatasi laju perubahan pedal gas dan rem
            # Rem harus responsif (+25% per step) agar mobil sempat melambat sebelum tikungan 90°
            max_brake_rise = 0.25   # Kenaikan rem responsif +25% per step
            max_brake_fall = 0.25   # Pelepasan rem max -25% per step
            if target_brake > self.prev_brake:
                limited_brake = min(target_brake, self.prev_brake + max_brake_rise)
            else:
                limited_brake = max(target_brake, self.prev_brake - max_brake_fall)

            max_throttle_rise = 0.15  # Akselerasi gas bertahap max +15% per step
            max_throttle_fall = 0.30  # Pelepasan gas cepat saat mendekati tikungan
            if target_throttle > self.prev_throttle:
                limited_throttle = min(target_throttle, self.prev_throttle + max_throttle_rise)
            else:
                limited_throttle = max(target_throttle, self.prev_throttle - max_throttle_fall)

            # =================================================================
            # PENGAWAS KESELAMATAN TERPADU:
            # Polisi 1: Laning Shield (CBE - Paramount Constraint)
            # Polisi 2: Stability Shield (CLS - Secondary Constraint)
            # =================================================================
            (
                cmd_steering,
                cmd_throttle,
                cmd_brake,
                cbe_intervened,
                cls_intervened
            ) = self.safety_shield.enforce(
                raw_steering=raw_steering,
                raw_throttle=limited_throttle,
                raw_brake=limited_brake,
                d_curb_phys_right=d_curb_phys_right,
                d_curb_phys_left=d_curb_phys_left,
                d_curb_eff=d_curb_eff,
                speed_ms=speed_ms,
                e_yaw=e_yaw,
                curv=curv,
                v_target_ms=v_target
            )

            if cbe_intervened or cls_intervened:
                shield_intervened = True

            self.prev_throttle = cmd_throttle
            self.prev_brake = cmd_brake

        # Cetak log diagnostik aksi kemudi di terminal saat berada di tikungan
        if abs(curv) > 0.03 or abs(cmd_steering) > 0.15 or d_corridor_phys < 0.50:
            cbe_str = " [CBE!]" if cbe_intervened else ""
            cls_str = " [CLS!]" if cls_intervened else ""
            t_str = datetime.now().strftime("%H:%M:%S")
            print(f"  [{t_str}] [CORNER LOG] Stanley: {raw_steering:+.3f} | Final: {cmd_steering:+.3f}{cbe_str}{cls_str} | e_yaw: {e_yaw:+.2f} rad | d_phys_R: {d_curb_phys_right:.2f}m | Spd: {speed_ms*3.6:4.1f} km/h (V_tgt: {v_target*3.6:4.1f})")

        # JAMINAN ANTI-MUNDUR (Zero Reverse Lockout):
        # Simulator BeamNG dalam mode 'arcade' mengartikan input pedal rem (brake)
        # saat mobil diam atau lambat (< 0.5 m/s) sebagai perintah pedal gas mundur (Reverse Throttle)!
        # Kunci rem hidrolik = 0.0 saat kecepatan sangat rendah agar simulator 100% melaju maju (Drive).
        if speed_ms < 0.5 and not self.shield_latched:
            cmd_brake = 0.0
        elif self.shield_latched and speed_ms < 0.3:
            cmd_brake = 0.0
            cmd_parkingbrake = 1.0

        # Terapkan kontrol ke kendaraan ego
        self.ego.control(
            throttle=cmd_throttle,
            brake=cmd_brake,
            steering=cmd_steering,
            parkingbrake=cmd_parkingbrake
        )

        # Kontrol mobil rintangan (jika ada dan statis, kunci agar tidak mundur)
        if self.has_obstacle and self.obstacle is not None and not self.is_moving_obstacle:
            self.obstacle.control(gear=0, throttle=0.0, brake=1.0, steering=0.0, parkingbrake=1.0)

        self.bng.control.step(15)

        next_obs = self._get_obs()
        new_speed, new_dist, _ = next_obs[0], next_obs[1], next_obs[2]

        # 4. Fungsi Reward: Mengikuti Profil Kecepatan Kinematis
        target_speed_ms = v_target
        if not self.has_obstacle or new_dist > (d_safe + 15.0):
            reward_speed = -abs(new_speed - target_speed_ms) / max(target_speed_ms, 1.0)
        else:
            target_approach_speed = max(0.0, min(v_target, target_speed_ms * ((new_dist - self.d_margin) / 25.0)))
            reward_speed = -abs(new_speed - target_approach_speed) / max(target_speed_ms, 1.0)

        reward_shield_penalty = -2.0 if (shield_intervened and speed_error > 2.0) else 0.0
        reward_lateral_penalty = -0.8 * min(abs(e_lat), 2.5)

        # Early Warning Penalti Kedekatan Trotoar Berdasarkan Fusi Multi-Modal (Visual YOLO + Koridor Fisik)
        # Menjamin agen tetap berhati-hati meskipun modul visi mengalami false-positive pada trotoar abu-abu
        reward_curb_warning = 0.0
        if d_curb_eff < 0.60 and new_speed <= 15.0:
            reward_curb_warning = -4.0 * (0.60 - d_curb_eff)

        # Progress Reward Shaping sepanjang kurva Frenet (Mencegah Exploration Bottleneck di Awal)
        curr_wp_dist = float(self.df_wp['distance_m'].iloc[closest_idx])
        prev_wp_dist = getattr(self, 'prev_distance_m', curr_wp_dist)
        delta_progress_m = max(0.0, curr_wp_dist - prev_wp_dist)
        self.prev_distance_m = curr_wp_dist

        reward_progress = 1.0 * min(delta_progress_m, 2.5)  # Imbalan positif gerak maju sepanjang rute
        reward_heading = -0.4 * (e_yaw ** 2)                # Panduan kelurusan sudut hadap kontinu

        # 1. PENALTI PING-PONG / STEER RATE (Peredam Zigzag & Osilasi Lateral)
        delta_steer = cmd_steering - getattr(self, 'last_steer_action', cmd_steering)
        steer_rate_penalty = -0.25 * (delta_steer ** 2)
        self.last_steer_action = cmd_steering

        reward = (
            reward_speed
            + reward_shield_penalty
            + reward_lateral_penalty
            + reward_curb_warning
            + reward_progress
            + reward_heading
            + steer_rate_penalty
        )

        terminated = False
        truncated = False
        is_collision = False
        is_collision_obstacle = False
        is_collision_structural = False
        is_safe_stop = False
        is_curb_climb = False
        is_off_track = False
        is_stalled = False
        termination_reason = "RUNNING"

        # A. Tabrakan dengan Mobil Rintangan di Depan (Prioritas 1: Bahaya Fatal)
        if self.has_obstacle and new_dist < 0.8 and new_speed > 0.8:
            reward -= 20.0  # Penalti softened: tabrakan fatal rintangan (mencegah Value Loss shock)
            terminated = True
            is_collision = True
            is_collision_obstacle = True
            termination_reason = "OBSTACLE_COLLISION"

        # B. Cek Sensor Kerusakan Fisik Kendaraan (Benturan Keras Deformasi Struktur Sasis > 50)
        damage_val = 0.0
        if hasattr(self, 'damage') and getattr(self.damage, 'data', None):
            damage_val = float(self.damage.data.get('damage', 0.0))
        
        if not terminated and damage_val > 50.0:
            reward -= 18.0  # Penalti softened: benturan keras tiang / dinding beton
            terminated = True
            is_collision = True
            is_collision_structural = True
            termination_reason = "STRUCTURAL_CRASH"

        # C. Wasit Lingkungan Fisik (Ground-Truth Multi-Modal Observer):
        # Menerapkan Gerbang Logika Konjungtif (Gerbang AND) Bebas Kesaksian Palsu:
        # 1. Saksi Geometri (OBB Tapak Roda): Roda fisik terluar (Front/Rear) melintasi batas lebar lajur aspal (> 1.72m)
        # 2. Saksi Fisika Sasis (Roll Elevasi Undakan): Sasis mengalami kemiringan tajam akibat menaiki undakan beton (|roll| >= 5.0°)
        # Jika ban masih berada di atas aspal (incursion <= 1.70m), bodi miring akibat transfer beban sentrifugal
        # dibebaskan secara adil dan tidak divonis sebagai pelanggaran melindas trotoar.
        wheel_inc_r = max(bbox.get("front_right", e_lat + 0.93), bbox.get("rear_right", e_lat + 0.93))
        wheel_inc_l = max(-bbox.get("front_left", e_lat - 0.93), -bbox.get("rear_left", e_lat - 0.93))

        roll_deg = getattr(self, 'last_roll_deg', 0.0)
        is_curb_climb_right = (wheel_inc_r > 1.72 and roll_deg <= -5.0)
        is_curb_climb_left  = (wheel_inc_l > 1.72 and roll_deg >= 5.0)
        is_curb_climb = (is_curb_climb_right or is_curb_climb_left)

        if not terminated and is_curb_climb:
            reward -= 14.0  # Penalti softened: pelanggaran trotoar pejalan kaki
            terminated = True
            termination_reason = "CURB_CLIMB"

        # D. Batas Fisik Keluar Aspal Jalan (Adaptive Off-Track Bound)
        # Di pelataran terbuka SPBU (< 45m), toleransi diperlebar menjadi 3.8m agar eksplorasi bermanuver keluar tidak terputus dini.
        # Di jalan raya bertrotoar/bermarka normal (>= 45m), batas ketat 2.0m diberlakukan.
        curr_dist_m = float(self.df_wp['distance_m'].iloc[closest_idx])
        e_lat_limit = 3.8 if curr_dist_m < 45.0 else 2.0

        if not terminated and abs(e_lat) > e_lat_limit:
            reward -= 10.0  # Penalti softened: keluar marka/badan jalan tanpa benturan
            terminated = True
            is_off_track = True
            termination_reason = "OFF_TRACK"

        # E. Deteksi Mobil Tersangkut / Mogok Menabrak Trotoar (Stuck against curb/wall/pole)
        # Jika mobil diinstruksikan maju (gas > 8%) tetapi kecepatan terhenti (< 0.3 m/s) selama 6 step (~0.9 detik):
        if self.step_count > 6 and new_speed < 0.3 and cmd_throttle > 0.08 and cmd_brake < 0.2:
            self.stalled_steps = getattr(self, 'stalled_steps', 0) + 1
            if self.stalled_steps >= 6 and not terminated:
                reward -= 10.0  # Penalti softened: kegagalan operasional gerak
                terminated = True
                is_stalled = True
                termination_reason = "VEHICLE_STUCK"
        else:
            self.stalled_steps = 0

        # F. Berhenti aman di depan rintangan (Uji Safety Stop Collision Avoidance)
        if not terminated and self.has_obstacle and self.step_count > 5 and new_speed < 0.2:
            if new_dist <= 15.0:
                if 2.0 <= new_dist <= 6.0:
                    reward += 25.0 if not self.shield_latched else 15.0
                else:
                    reward += 15.0 if not self.shield_latched else 8.0
                terminated = True
                is_safe_stop = True
                termination_reason = "SAFE_STOP"
            elif self.shield_latched:
                reward += 5.0
                terminated = True
                is_safe_stop = True
                termination_reason = "SAFE_STOP"

        # G. Berhasil menyelesaikan seluruh rute CSV (Finish Line + Conditional Time-Arrival Bonus)
        is_route_completed = False
        if not terminated and closest_idx >= len(self.df_wp) - 5:
            base_finish_reward = 100.0
            # Hitung rasio efisiensi waktu penyelesaian rute (semakin cepat & sedikit step, makin besar bonus +0 s.d. +50)
            time_efficiency_ratio = max(0.0, float(self.max_steps - self.step_count) / float(self.max_steps))
            time_bonus = 50.0 * time_efficiency_ratio
            reward += (base_finish_reward + time_bonus)
            terminated = True
            is_route_completed = True
            termination_reason = "ROUTE_COMPLETED"
            print(f"  [GOAL FINISH!] Rute Selesai Sukses! Step: {self.step_count}/{self.max_steps} | Base: +{base_finish_reward:.1f} | Bonus Waktu: +{time_bonus:.2f}")

        if self.step_count >= self.max_steps and not terminated:
            if not self.has_obstacle and abs(e_lat) <= 1.8:
                reward += 25.0  # Bonus sukses jelajah rute tanpa keluar jalur
            truncated = True
            termination_reason = "MAX_STEPS_REACHED"

        bbox = getattr(self.wm, 'last_bbox_errors', {})

        info = {
            "shield_triggered": shield_intervened,
            "cbe_intervened": cbe_intervened,
            "cls_intervened": cls_intervened,
            "cbe_lateral_triggered": cbe_intervened,
            "d_safe": d_safe,
            "dist_actual": new_dist,
            "collision": is_collision,
            "collision_obstacle": is_collision_obstacle,
            "collision_structural": is_collision_structural,
            "curb_climb": is_curb_climb,
            "off_track": is_off_track,
            "stuck": is_stalled,
            "termination_reason": termination_reason,
            "safe_stop": is_safe_stop,
            "route_completed": is_route_completed,
            "e_lat": e_lat,
            "e_lat_rear_right": bbox.get("rear_right", e_lat),
            "ref_offset": getattr(self.wm, 'last_ref_offset', 0.0),
            "w_rear": getattr(self.wm, 'last_w_rear', 0.0),
            "d_curb_lidar": getattr(self, 'last_d_curb_lidar', 4.0),
            "d_curb_phys": d_corridor_phys,
            "d_curb_visual": d_curb_visual,
            "d_curb_eff": d_curb_eff,
            "d_curb_yolo": getattr(self, 'last_d_curb_yolo', 4.0),
            "yolo_hud_frame": getattr(self, 'last_yolo_hud_frame', None),
            "roll_deg": roll_deg,
            "steering": cmd_steering,
            "steer_rate_penalty": steer_rate_penalty,
            "v_target": v_target,
            "v_curve_safe": v_target,
            "curv": curv,
            "throttle": cmd_throttle,
            "brake": cmd_brake,
            "progress_m": float(self.df_wp['distance_m'].iloc[closest_idx] - self.df_wp['distance_m'].iloc[0]),
            "total_route_m": float(self.df_wp['distance_m'].iloc[-1] - self.df_wp['distance_m'].iloc[0])
        }

        return next_obs, reward, terminated, truncated, info

    def close(self):
        if getattr(self, 'camera', None):
            try:
                self.camera.remove()
            except Exception:
                pass
        if getattr(self, 'us_front_right', None):
            try:
                self.us_front_right.remove()
            except Exception:
                pass
        if getattr(self, 'us_rear_right', None):
            try:
                self.us_rear_right.remove()
            except Exception:
                pass
        if self.bng:
            self.bng.close()
