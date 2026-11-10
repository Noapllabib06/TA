"""
Skrip Perekaman Dataset Otomatis Kamera Semantik (RGB + Anotasi) BeamNG.tech
--------------------------------------------------------------------------
Mengemudikan kendaraan secara otonom di rute Suburban 'east_coast_usa'
dan merekam pasangan citra fotorealistik RGB + citra anotasi semantik (Ground-Truth).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from waypoint_manager import WaypointManager

BNG_HOME = r"C:\BeamNG\BeamNG.tech.v0.39.4.0"
CSV_PATH = os.path.join(os.path.dirname(__file__), 'waypoints_suburban_town.csv')
RAW_DIR = os.path.join(os.path.dirname(__file__), "dataset_raw")
IMG_DIR = os.path.join(RAW_DIR, "images")
ANN_DIR = os.path.join(RAW_DIR, "annotations")

def main():
    print("===========================================================")
    print("      KOLEKTOR DATASET OTOMATIS: RGB + SEMANTIC MASK       ")
    print("===========================================================")

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(ANN_DIR, exist_ok=True)

    # 1. Muat Waypoints Rute Suburban Target
    df_wp = pd.read_csv(CSV_PATH)
    waypoints = df_wp[['x', 'y', 'z']].values
    wm = WaypointManager(waypoints=waypoints, k_e=0.7, k_soft=1.5, target_cruise_kmh=35.0)

    start_pos, start_rot = WaypointManager.calculate_start_pose(df_wp)

    # 2. Inisialisasi Simulator BeamNG.tech
    print(f"-> Membuka BeamNG.tech pada '{BNG_HOME}'...")
    bng = BeamNGpy('localhost', 64256, home=BNG_HOME)
    bng.open(launch=True)

    scenario = Scenario('east_coast_usa', 'data_collector_curb')
    ego = Vehicle('ego_vehicle', model='etk800')
    scenario.add_vehicle(ego, pos=start_pos, rot_quat=start_rot)

    print("-> Memuat skenario dan me-render kendaraan...")
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(2.0)  # Pemanasan engine rendering DirectX 12

    ego.set_shift_mode('arcade')

    # 3. Konfigurasi Kamera Depan (RGB + Semantic Annotation)
    # Posisi di ujung bemper depan (pos=(0, -2.15, 0.85)), bebas dari halangan kap mobil (hood)
    # Sudut menunduk ~12.5 derajat langsung menyorot aspal dan bibir trotoar di depan
    print("-> Mengaktifkan Kamera Bemper Depan (640x480, FOV 70°, Bebas Halangan Kap Mesin)...")
    cam = Camera(
        name='front_cam_collector',
        bng=bng,
        vehicle=ego,
        requested_update_time=0.1,
        pos=(0.0, -2.15, 0.85),
        dir=(0.0, -1.0, -0.22),  # Pitch down ~12.5° langsung ke jalan & trotoar
        up=(0.0, 0.0, 1.0),
        resolution=(640, 480),
        field_of_view_y=70,
        near_far_planes=(0.1, 120.0),
        is_using_shared_memory=False,
        is_render_colours=True,
        is_render_annotations=True,
        is_render_depth=False,
        is_visualised=False
    )

    TOTAL_SAMPLES = 800
    SAMPLE_INTERVAL_STEPS = 5   # Ambil sampel tiap 5 langkah simulasi (~1.5 meter pergerakan)
    collected_count = 0

    print(f"\n--- Memulai Perekaman Dataset ({TOTAL_SAMPLES} Frame) di Rute Suburban ---")
    print(f"{'Frame ID':<12} | {'Kecepatan':<12} | {'Posisi Jalur (e_lat)':<22} | {'Progress'}")
    print("-" * 65)

    try:
        step = 0
        while collected_count < TOTAL_SAMPLES:
            step += 1
            
            # Navigasi Lateral Otomatis Stanley agar mobil menyusuri seluruh kontur jalan & tikungan
            ego.sensors.poll()
            pos = np.array(ego.state.get('pos', (0, 0, 0)))
            dir_vec = np.array(ego.state.get('dir', (1, 0, 0)))
            vel_vec = np.array(ego.state.get('vel', (0, 0, 0)))
            speed_ms = float(np.linalg.norm(vel_vec))
            ego_yaw = float(np.arctan2(dir_vec[1], dir_vec[0]))

            steer_cmd, e_lat, _ = wm.compute_steering(pos, ego_yaw, speed_ms)

            # Kendali Kecepatan Jelajah Santai ~25-30 km/jam agar kualitas citra tajam (tanpa motion blur)
            target_v = 7.5  # m/s (~27 km/jam)
            throttle_cmd = 0.30 if speed_ms < target_v else 0.05
            brake_cmd = 0.25 if speed_ms > (target_v + 1.5) else 0.0

            ego.control(throttle=throttle_cmd, brake=brake_cmd, steering=steer_cmd)
            bng.control.step(SAMPLE_INTERVAL_STEPS)

            # Ambil data sensor kamera
            cam_data = cam.poll()
            if not cam_data or 'colour' not in cam_data or 'annotation' not in cam_data:
                continue

            rgb_img = cam_data['colour']
            ann_img = cam_data['annotation']

            if rgb_img is None or ann_img is None:
                continue

            # Simpan pasangan citra
            file_id = f"frame_{collected_count:05d}"
            rgb_path = os.path.join(IMG_DIR, f"{file_id}.jpg")
            ann_path = os.path.join(ANN_DIR, f"{file_id}.png")

            rgb_img.convert('RGB').save(rgb_path, quality=95)
            ann_img.convert('RGB').save(ann_path)

            collected_count += 1

            if collected_count % 25 == 0 or collected_count == TOTAL_SAMPLES:
                pct = (collected_count / TOTAL_SAMPLES) * 100.0
                print(f"{file_id:<12} | {speed_ms*3.6:5.1f} km/h   | e_lat: {e_lat:+5.2f} m        | [{collected_count}/{TOTAL_SAMPLES} ({pct:4.1f}%)]")

    except KeyboardInterrupt:
        print("\nPerekaman dihentikan oleh pengguna.")
    finally:
        try:
            cam.remove()
        except Exception:
            pass
        bng.close()
        print(f"\n[SELESAI] Total {collected_count} pasangan citra tersimpan di folder '{RAW_DIR}'.")

if __name__ == "__main__":
    main()

