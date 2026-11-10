from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Lidar, Electrics
import numpy as np
import time

# 1. Konfigurasi Path dan Inisialisasi Simulator
BNG_HOME = r"C:\BeamNG\BeamNG.tech.v0.39.4.0"

bng = BeamNGpy('localhost', 64256, home=BNG_HOME)
bng.open(launch=True)

scenario = Scenario('automation_test_track', 'test_actuation_telemetry')

# 2. Spawn Ego Vehicle
ego_start_pos = (489.20, 178.71, 132.47)
ego_rot = (0, 0, 0.7071, 0.7071)

ego = Vehicle('ego_vehicle', model='etk800')
scenario.add_vehicle(ego, pos=ego_start_pos, rot_quat=ego_rot)

# 3. Spawn Lead Vehicle (Rintangan statis 50 meter di depan lajur lintasan)
obstacle_start_pos = (ego_start_pos[0] - 50.0, ego_start_pos[1], ego_start_pos[2])
obstacle = Vehicle('obstacle_box', model='etkc')
scenario.add_vehicle(obstacle, pos=obstacle_start_pos, rot_quat=ego_rot)

scenario.make(bng)
bng.scenario.load(scenario)
bng.scenario.start()

ego.set_shift_mode('arcade')

# 4. Pasang Sensor Electrics (Membaca throttle & brake aktual)
electrics = Electrics()
ego.attach_sensor('electrics', electrics)

# 5. Pasang Sensor LiDAR di Bemper Depan (Sumbu lokal BeamNG: -Y adalah arah depan)
lidar = Lidar(
    name='front_lidar',
    bng=bng,
    vehicle=ego,
    requested_update_time=0.05,
    is_using_shared_memory=False,
    pos=(0.0, -2.3, 0.6),           # Ujung bemper depan
    dir=(0.0, -1.0, 0.0),           # Menembak lurus ke depan kendaraan
    up=(0.0, 0.0, 1.0),
    vertical_resolution=16,
    horizontal_angle=40,
    vertical_angle=12,
    max_distance=60.0,
    is_visualised=True
)

print("\n--- Sistem Siap: Uji Safety Shield + Handbrake Lock ---")
print(f"{'Step':<5} | {'Speed':<10} | {'Gas':<5} | {'Rem':<5} | {'LiDAR (m)':<10} | {'GT Dist':<10} | {'Pitch':<7} | {'Status'}")
print("-" * 80)

# Parameter Kontrol & Ambang Batas
cmd_throttle = 0.5
cmd_brake = 0.0
cmd_parkingbrake = 0.0
aeb_triggered = False

JARAK_INTERVENSI_SAFE = 18.0  # Threshold intervensi pengereman darurat (meter)
LANE_HALF_WIDTH = 1.7        # Setengah lebar lajur jalur lalu lintas (meter)

for step in range(35):
    # Kirim perintah kendali ke aktuator ego vehicle
    ego.control(
        throttle=cmd_throttle,
        steering=0.0,
        brake=cmd_brake,
        parkingbrake=cmd_parkingbrake
    )
    bng.control.step(15)
    
    # Perbarui data seluruh sensor
    ego.sensors.poll()
    obstacle.sensors.poll()
    
    # 1. Telemetri Kinematika & Orientasi
    ego_pos = np.array(ego.state.get('pos', (0, 0, 0)))
    dir_vec = np.array(ego.state.get('dir', (1, 0, 0)))
    vel_vec = np.array(ego.state.get('vel', (0, 0, 0)))
    obs_pos = np.array(obstacle.state.get('pos', (0, 0, 0)))
    
    speed_kmh = np.linalg.norm(vel_vec) * 3.6
    pitch_deg = np.degrees(np.arcsin(np.clip(dir_vec[2], -1.0, 1.0)))
    
    # Ground-truth jarak fisik antar bodi kendaraan
    gt_dist = np.linalg.norm(obs_pos - ego_pos) - 4.5
    
    actual_gas = electrics.data.get('throttle', 0.0)
    actual_brake = electrics.data.get('brake', 0.0)
    
    # 2. Vektor Basis Lokal Kendaraan di Bidang Horizontal
    fwd_unit = np.array([dir_vec[0], dir_vec[1], 0.0])
    fwd_unit /= np.linalg.norm(fwd_unit)
    left_unit = np.array([-fwd_unit[1], fwd_unit[0], 0.0])
    
    bumper_pos = ego_pos + (fwd_unit * 2.3)
    
    # 3. Pembacaan Point Cloud & Filtering Koridor Jalur (ROI)
    data = lidar.poll()
    points = data.get('pointCloud', None)
    min_dist = float('inf')
    
    if points is not None and len(points) > 0:
        rel_points = points - bumper_pos
        dist_fwd = np.dot(rel_points[:, :2], fwd_unit[:2])
        dist_lat = np.dot(rel_points[:, :2], left_unit[:2])
        dist_vert = points[:, 2] - ego_pos[2]
        
        # Filter objek di dalam koridor lajur: abaikan aspal, langit, dan guardrail samping
        roi_mask = (
            (dist_fwd > 0.5) & 
            (np.abs(dist_lat) <= LANE_HALF_WIDTH) & 
            (dist_vert >= 0.20) & (dist_vert <= 2.2)
        )
        
        lane_obstacles = dist_fwd[roi_mask]
        if len(lane_obstacles) > 0:
            min_dist = np.min(lane_obstacles)

    # 4. Logika Safety Shield AEB & Penguncian Rem Tangan
    status = "CRUISE (DRL)"
    
    if min_dist <= JARAK_INTERVENSI_SAFE or gt_dist <= JARAK_INTERVENSI_SAFE:
        aeb_triggered = True

    if aeb_triggered:
        if speed_kmh > 1.0:
            # Fase Deselerasi Keras
            status = "SHIELD ACTIVE (AEB)"
            cmd_throttle = 0.0
            cmd_brake = 1.0
            cmd_parkingbrake = 0.0
        else:
            # Fase Berhenti Aman: Lepas pedal rem dan kunci roda mekanis
            status = "STOPPED SECURE"
            cmd_throttle = 0.0
            cmd_brake = 0.0
            cmd_parkingbrake = 1.0

    lidar_str = f"{min_dist:6.2f} m" if min_dist != float('inf') else "  CLEAR  "
    print(f"{step+1:02d}   | {speed_kmh:6.2f} km/h | {actual_gas:4.2f} | {actual_brake:4.2f} | {lidar_str:<10} | {gt_dist:6.2f} m   | {pitch_deg:5.2f}° | {status}")
    time.sleep(0.04)

bng.close()
print("\nPengujian selesai.")