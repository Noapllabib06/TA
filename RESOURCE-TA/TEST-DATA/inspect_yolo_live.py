"""
Live Visual Inspector: Modul Persepsi Visual YOLOv8-seg di Simulator BeamNG.tech
-------------------------------------------------------------------------------
Skrip ini dibuat khusus untuk memvisualisasikan secara langsung (real-time):
1. Citra mentah dari kamera bemper depan kendaraan.
2. Prediksi Masker Segmentasi Semantik (Poligon Berwarna):
   - Warna Cyan/Biru  : Permukaan Jalan (Road)
   - Warna Magenta/Merah: Bibir Trotoar (Curb)
3. Kotak Pembatas (Bounding Box) dan Nilai Keyakinan (Confidence Score) deteksi.
4. Dashboard HUD Telemetri Spasial Numerik:
   - Jarak Lateral ke Trotoar Kanan (d_curb)
   - Deviasi Garis Tengah Jalan (e_y)
   - Sudut Arah Kurvatur Jalan (e_psi)
   - Kecepatan & Latensi Inferensi (FPS)

Tekan tombol 'q' pada jendela visualisasi OpenCV untuk menghentikan inspeksi.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import pandas as pd

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from waypoint_manager import WaypointManager
from yolo_spatial_extractor import YOLOSpatialExtractor

def main():
    parser = argparse.ArgumentParser(description="Live Visual Inspector YOLOv8-seg di BeamNG.tech")
    parser.add_argument("--speed", type=float, default=30.0, help="Target kecepatan jelajah inspeksi (km/jam, default: 30)")
    parser.add_argument("--steps", type=int, default=500, help="Jumlah langkah inspeksi (default: 500)")
    args = parser.parse_args()

    print("===========================================================")
    print("      LIVE VISUAL INSPECTOR: YOLOV8-SEG MATA OTONOM        ")
    print("===========================================================")

    # 1. Inisialisasi Model YOLOv8-seg & Ekstraktor Spasial
    extractor = YOLOSpatialExtractor()
    print("-> Ekstraktor Spasial YOLOv8-seg SIAP!")

    # 2. Inisialisasi Simulator BeamNG.tech
    bng_home = r"C:\BeamNG\BeamNG.tech.v0.39.4.0"
    print(f"-> Membuka BeamNG.tech pada '{bng_home}'...")
    bng = BeamNGpy('localhost', 64256, home=bng_home)
    bng.open(launch=True)

    # 3. Muat Skenario Rute Sub-Urban East Coast USA
    scenario = Scenario('east_coast_usa', 'yolo_live_inspector')
    csv_path = os.path.join(os.path.dirname(__file__), 'waypoints_suburban_town.csv')
    df_wp = pd.read_csv(csv_path)
    waypoints = df_wp[['x', 'y', 'z']].values

    ego_start_pos, ego_start_rot = WaypointManager.calculate_start_pose(df_wp)
    ego = Vehicle('ego_vehicle', model='etk800')
    scenario.add_vehicle(ego, pos=ego_start_pos, rot_quat=ego_start_rot)

    print("-> Memuat skenario dan me-render kendaraan...")
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(1.5)

    ego.set_shift_mode('arcade')

    # 4. Pasang Kamera Bemper Depan (Bebas Halangan Kap Mesin)
    print("-> Mengaktifkan Kamera Bemper Depan (640x480, FOV 70°)...")
    cam = Camera(
        name='front_inspector_cam',
        bng=bng,
        vehicle=ego,
        requested_update_time=0.08,
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
    print("-> Sensor Kamera berhasil terpasang dan aktif.")

    # 5. Inisialisasi Kontroler Stanley untuk Menjaga Mobil Tetap di Lajur Jalan
    wm = WaypointManager(waypoints, target_cruise_kmh=args.speed)

    print("\n--- MEMULAI LIVE INSPECTION VISUAL YOLOV8-SEG ---")
    print("Perhatikan jendela 'Live Visual: YOLOv8-seg Perception HUD' yang muncul di layar!")
    print("Tekan 'q' pada jendela OpenCV kapan saja untuk keluar.\n")
    print(f"{'Step':<5} | {'Kecepatan':<10} | {'d_curb (Trotoar)':<18} | {'e_y (Tengah)':<14} | {'e_psi (Heading)':<16} | {'Status Visual'}")
    print("-" * 90)

    cv2.namedWindow("Live Visual: YOLOv8-seg Perception HUD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Visual: YOLOv8-seg Perception HUD", 800, 600)

    try:
        prev_throttle = 0.0
        prev_brake = 0.0

        for step in range(1, args.steps + 1):
            ego.sensors.poll()
            ego_pos = np.array(ego.state.get('pos', (0, 0, 0)))
            dir_vec = np.array(ego.state.get('dir', (1, 0, 0)))
            vel_vec = np.array(ego.state.get('vel', (0, 0, 0)))
            ego_yaw = float(np.arctan2(dir_vec[1], dir_vec[0]))
            speed_ms = float(np.linalg.norm(vel_vec))
            speed_kmh = speed_ms * 3.6

            # Kendali kemudi stabil mengikuti jalur rekaman
            steer_cmd, e_lat, _ = wm.compute_steering(ego_pos, ego_yaw, speed_ms)

            # Kendali kecepatan jelajah konstan
            target_speed_ms = args.speed / 3.6
            v_err = speed_ms - target_speed_ms
            if v_err > 0.5:
                cmd_throttle = 0.0
                cmd_brake = float(np.clip(v_err / 3.0, 0.1, 0.5))
            else:
                cmd_brake = 0.0
                cmd_throttle = float(np.clip(0.25 - 0.20 * (v_err / 0.5), 0.10, 0.45))

            ego.control(throttle=cmd_throttle, brake=cmd_brake, steering=steer_cmd)
            bng.control.step(12)

            # Polling kamera dan proses YOLOv8-seg
            cam_data = cam.poll()
            if cam_data and 'colour' in cam_data and cam_data['colour'] is not None:
                img_np = np.array(cam_data['colour'].convert('RGB'))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Ekstraksi metrik numerik spasial
                metrics = extractor.extract_metrics(img_bgr)

                # Render frame HUD teranotasi lengkap (Bounding box + Masker Poligon + Telemetri)
                annotated_hud = extractor.get_annotated_hud()

                if annotated_hud is not None:
                    # Tambahkan info kecepatan mobil ego pada HUD
                    cv2.putText(annotated_hud, f"Ego Speed: {speed_kmh:4.1f} km/h | Step: {step:03d}/{args.steps}",
                                (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)

                    cv2.imshow("Live Visual: YOLOv8-seg Perception HUD", annotated_hud)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nPengguna menghentikan inspeksi visual.")
                        break

                curb_status = "CURB DETECTED" if metrics.is_curb_detected else "NO CURB"
                print(f"{step:04d}  | {speed_kmh:5.1f} km/h | {metrics.d_curb:+5.2f} m ({curb_status:<13}) | {metrics.e_y:+5.2f} m     | {np.degrees(metrics.e_psi):+5.1f} deg      | Latency: {metrics.inference_ms:4.1f} ms")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInspeksi dihentikan secara manual.")
    finally:
        cv2.destroyAllWindows()
        try:
            cam.remove()
        except Exception:
            pass
        bng.close()
        print("Skenario dan simulator ditutup dengan sukses.")

if __name__ == "__main__":
    main()

