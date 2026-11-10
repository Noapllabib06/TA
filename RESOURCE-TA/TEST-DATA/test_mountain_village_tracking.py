import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from beamngpy import BeamNGpy, Scenario, Vehicle

from waypoint_manager import WaypointManager

BNG_HOME = r"C:\BeamNG\BeamNG.tech.v0.39.4.0"
WAYPOINT_CSV = "waypoints_italy_mountain.csv"
DEFAULT_START_OFFSET_M = 45.0

def main():
    parser = argparse.ArgumentParser(description="Uji Kendali Lateral Mountain Village (Stanley)")
    parser.add_argument("--offset", type=float, default=DEFAULT_START_OFFSET_M,
                        help="Jarak awal (meter) yang dipotong langsung dari CSV (default: 45.0 m)")
    args, _ = parser.parse_known_args()

    print("===========================================================")
    print("    UJI KENDALI LATERAL OTONOM: MOUNTAIN (Italy Village)   ")
    print("===========================================================")
    
    if not os.path.exists(WAYPOINT_CSV):
        print(f"Error: Berkas {WAYPOINT_CSV} tidak ditemukan!")
        return

    # 1. Muat Waypoints Italy Mountain Village
    df_raw = pd.read_csv(WAYPOINT_CSV)
    df_wp = df_raw[df_raw['distance_m'] >= args.offset].reset_index(drop=True)
    waypoints = df_wp[['x', 'y', 'z']].values
    
    start_dist_actual = df_wp['distance_m'].iloc[0]
    dist_total = df_wp['distance_m'].iloc[-1] - start_dist_actual
    print(f"-> Melompat langsung ke meter {start_dist_actual:.1f} dari CSV (melewati {args.offset:.1f} m awal).")
    print(f"-> Memuat {len(waypoints)} waypoint jalur berliku (Sisa Rute: {dist_total:.1f} m)")

    # Inisialisasi WaypointManager dengan Stanley Controller
    wm = WaypointManager(waypoints=waypoints, k_e=0.75, k_soft=1.5)

    # Hitung posisi dan orientasi hadap awal analitik agar selaras dengan jalan (e_yaw = 0.0°)
    start_pos, start_rot = WaypointManager.calculate_start_pose(df_wp)
    print(f"-> Garis Start Baru di Jalan Pedesaan : ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")
    print(f"-> Orientasi Spawn Selaras Jalur      : {start_rot}")

    # 2. Inisialisasi Simulator & Peta italy
    print("-> Membuka instance BeamNG.tech pada peta 'italy'...")
    bng = BeamNGpy('localhost', 64256, home=BNG_HOME)
    bng.open(launch=True)

    scenario = Scenario('italy', 'test_mountain_village_tracking')
    ego = Vehicle('ego_vehicle', model='etk800')
    scenario.add_vehicle(ego, pos=start_pos, rot_quat=start_rot)
    
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()

    ego.set_shift_mode('arcade')

    print("\n--- Memulai Pengujian Otonom Jalur Berliku (100 Step ~ 300 meter) ---")
    print(f"{'Step':<5} | {'Speed':<10} | {'Steering':<9} | {'e_lat (m)':<10} | {'e_yaw':<8} | {'Pitch':<8} | {'Progress':<10} | {'Status'}")
    print("-" * 85)

    CRUISE_SPEED_MS = 11.0  # Kecepatan jelajah ~40 km/jam
    A_LAT_MAX = 2.8         # Batas kenyamanan percepatan sentripetal (m/s^2)
    errors_lat = []
    errors_yaw = []
    
    for step in range(100):
        ego.sensors.poll()
        pos = np.array(ego.state.get('pos', (0, 0, 0)))
        dir_vec = np.array(ego.state.get('dir', (1, 0, 0)))
        vel_vec = np.array(ego.state.get('vel', (0, 0, 0)))

        speed_ms = float(np.linalg.norm(vel_vec))
        speed_kmh = speed_ms * 3.6
        pitch_deg = float(np.degrees(np.arcsin(np.clip(dir_vec[2], -1.0, 1.0))))
        ego_yaw = float(np.arctan2(dir_vec[1], dir_vec[0]))

        # 1. Kendali Lateral Otomatis (Stanley Controller)
        steering_cmd, e_lat, e_yaw = wm.compute_steering(pos, ego_yaw, speed_ms)
        errors_lat.append(abs(e_lat))
        errors_yaw.append(abs(np.degrees(e_yaw)))

        # 2. Kecepatan Adaptif terhadap Kelengkungan Tikungan
        closest_idx = wm.find_closest_index(pos)
        curv = wm.get_curvature(closest_idx, lookahead_steps=6)
        if abs(curv) > 0.01:
            target_speed_ms = max(4.0, min(CRUISE_SPEED_MS, np.sqrt(A_LAT_MAX / abs(curv))))
        else:
            target_speed_ms = CRUISE_SPEED_MS

        # 3. Kendali Longitudinal
        if speed_ms < target_speed_ms:
            throttle_cmd = 0.45
            brake_cmd = 0.0
        else:
            throttle_cmd = 0.05
            brake_cmd = 0.30 if speed_ms > (target_speed_ms + 1.2) else 0.10

        # Terapkan kontrol aktuator ke kendaraan
        ego.control(
            throttle=throttle_cmd,
            steering=steering_cmd,
            brake=brake_cmd,
            parkingbrake=0.0
        )
        bng.control.step(15)

        progress_m = df_wp['distance_m'].iloc[closest_idx] - df_wp['distance_m'].iloc[0]

        status = "TRACKING TEPAT" if abs(e_lat) < 0.35 else ("KOREKSI" if abs(e_lat) < 0.75 else "MELENCENG")
        print(f"{step+1:03d}   | {speed_kmh:5.1f} km/h | {steering_cmd:+.3f}    | {e_lat:+.3f} m   | {np.degrees(e_yaw):+5.1f}° | {pitch_deg:+5.1f}° | {progress_m:6.1f} m  | {status}")
        time.sleep(0.02)

    bng.close()

    # Evaluasi Statistik Kinerja Lateral
    mean_elat = float(np.mean(errors_lat))
    max_elat = float(np.max(errors_lat))
    mean_eyaw = float(np.mean(errors_yaw))

    print("\n" + "=" * 65)
    print("     HASIL EVALUASI KINERJA KENDALI LATERAL VILLAGE (STANLEY)    ")
    print("=" * 65)
    print(f"Rata-rata Deviasi Jalur (Mean e_lat)  : {mean_elat:.3f} meter (Target: < 0.35 m)")
    print(f"Deviasi Maksimal (Max e_lat)          : {max_elat:.3f} meter (Batas Lajur: 1.70 m)")
    print(f"Rata-rata Kesalahan Arah (Mean e_yaw) : {mean_eyaw:.2f} derajat (Target: < 5.0°)")
    
    if mean_elat < 0.35 and max_elat < 1.0:
        print("\n-> KESIMPULAN: LULUS PENGUJIAN! Kendali lateral sangat presisi dan stabil di lajur.")
    else:
        print("\n-> KESIMPULAN: Memerlukan penyesuaian parameter gain k_e.")
    print("=" * 65)

if __name__ == "__main__":
    main()

