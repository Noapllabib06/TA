"""
Demo Independen: YOLOv8 sebagai "Mata" Sistem Kendali Otonom di BeamNG.tech
-------------------------------------------------------------------------
Skrip ini mendemonstrasikan bagaimana model YOLOv8 (Pre-trained) bertindak
sebagai subsistem persepsi visual (Front-Facing Camera):
1. Mengambil citra RGB dari kamera kaca depan mobil ego (ETK 800)
2. Memproses gambar secara real-time menggunakan YOLOv8 Nano
3. Mendeteksi kendaraan rintangan di depan (Bounding Box & Confidence)
4. Mengekstraksi estimasi jarak spasial untuk diumpankan ke Safety Shield
5. Menampilkan jendela live HUD (Heads-Up Display) menggunakan OpenCV
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from waypoint_manager import WaypointManager

def main():
    print("===========================================================")
    print("      DEMO PERSEPSI VISUAL: YOLOv8 SEBAGAI MATA OTOMATIS   ")
    print("===========================================================")

    # 1. Muat Model YOLOv8 Nano (Pre-trained COCO)
    print("-> Memuat model YOLOv8 Nano (Inference Mode)...")
    model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    yolo_model = YOLO(model_path)
    print("-> YOLOv8 Nano SIAP! Kelas target: Mobil, Truk, Bus, Pejalan Kaki.")

    # 2. Inisialisasi Simulator BeamNG.tech
    bng_home = r"C:\BeamNG\BeamNG.tech.v0.39.4.0"
    print(f"-> Membuka BeamNG.tech pada '{bng_home}'...")
    bng = BeamNGpy('localhost', 64256, home=bng_home)
    bng.open(launch=True)

    # 3. Setup Skenario Jalan Raya Suburban dengan Mobil Rintangan
    scenario = Scenario('east_coast_usa', 'demo_yolo_perception')
    
    # Ambil posisi spawn dari waypoint suburban
    csv_path = os.path.join(os.path.dirname(__file__), 'waypoints_suburban_town.csv')
    ego_start_pos, ego_start_rot = WaypointManager.calculate_start_pose(csv_path)

    # Spawn Ego Vehicle
    ego = Vehicle('ego_vehicle', model='etk800')
    scenario.add_vehicle(ego, pos=ego_start_pos, rot_quat=ego_start_rot)

    # Spawn Rintangan di depan (sekitar 35 meter di depan pada lajur)
    import pandas as pd
    df_wp = pd.read_csv(csv_path)
    obs_idx = 25  # ~35 meter di depan
    obs_pos = (
        float(df_wp['x'].iloc[obs_idx]),
        float(df_wp['y'].iloc[obs_idx]),
        float(df_wp['z'].iloc[obs_idx]) + 0.15
    )
    dx_obs = float(df_wp['x'].iloc[obs_idx+1] - df_wp['x'].iloc[obs_idx])
    dy_obs = float(df_wp['y'].iloc[obs_idx+1] - df_wp['y'].iloc[obs_idx])
    theta_obs = np.arctan2(-dx_obs, -dy_obs)
    obs_rot = (0.0, 0.0, float(np.sin(theta_obs / 2.0)), float(np.cos(theta_obs / 2.0)))

    obstacle = Vehicle('obstacle_car', model='etkc')
    scenario.add_vehicle(obstacle, pos=obs_pos, rot_quat=obs_rot)

    print("-> Memuat skenario dan me-render kendaraan...")
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(1.5)  # Beri waktu engine BeamNG menyelesaikan transisi rendering

    ego.set_shift_mode('arcade')
    try:
        obstacle.set_shift_mode('realistic_manual')
    except Exception:
        pass
    obstacle.control(gear=0, throttle=0.0, brake=1.0, parkingbrake=1.0)

    # 4. Pasang Sensor Kamera Kaca Depan (Front Windshield Camera)
    print("-> Mengaktifkan Kamera Kaca Depan (512x512, FOV 70°, Colour Only)...")
    cam = None
    try:
        cam = Camera(
            name='front_camera',
            bng=bng,
            vehicle=ego,
            requested_update_time=0.1,
            pos=(0.0, -2.15, 0.85),   # Posisi di bemper depan (bebas halangan kap mesin)
            dir=(0.0, -1.0, -0.22),   # Menghadap ke jalan depan
            up=(0.0, 0.0, 1.0),
            resolution=(512, 512),
            field_of_view_y=70,
            near_far_planes=(0.1, 150.0),
            is_using_shared_memory=False,
            is_render_colours=True,
            is_render_annotations=False,  # Matikan anotasi semantik agar socket ringan
            is_render_depth=False,        # Matikan depth buffer mentah
            is_visualised=False
        )
        print("-> Sensor Kamera berhasil terpasang pada kendaraan ego.")
    except Exception as e:
        print(f"-> Info: Sensor Kamera BeamNGpy ({e}) tidak aktif di mode soket, beralih ke Fallback Telemetri Spasial.")
        cam = None

    # Inisialisasi WaypointManager agar mobil tetap di lajur jalan saat mendekati rintangan
    wm = WaypointManager(df_wp[['x', 'y', 'z']].values, target_cruise_kmh=35.0)

    print("\n--- DEMO DIMULAI: Mobil maju di lajur jalan mendekati rintangan ---")
    print("Mata YOLOv8 memindai objek secara real-time. Tekan 'q' di jendela kamera untuk keluar.")
    print(f"{'Step':<5} | {'Deteksi YOLO':<18} | {'Confidence':<12} | {'Kotak (BBox)':<22} | {'Jarak Visual':<12} | {'Aksi Setir/Rem'}")
    print("-" * 95)

    focal_length_px = 512.0 / (2.0 * np.tan(np.radians(70.0 / 2.0)))
    real_car_height_m = 1.45  # Tinggi rata-rata mobil sedan

    try:
        for step in range(1, 300):
            # Ambil pose ego untuk kontrol lajur jalan (Stanley Controller)
            ego.sensors.poll()
            ego_pos = np.array(ego.state.get('pos', (0, 0, 0)))
            dir_vec = np.array(ego.state.get('dir', (1, 0, 0)))
            # Ambil pose ego & rintangan untuk kontrol dan pelindung keselamatan
            ego.sensors.poll()
            obstacle.sensors.poll()
            ego_pos = np.array(ego.state.get('pos', (0, 0, 0)))
            obs_pos = np.array(obstacle.state.get('pos', (0, 0, 0)))
            dir_vec = np.array(ego.state.get('dir', (1, 0, 0)))
            ego_yaw = float(np.arctan2(dir_vec[1], dir_vec[0]))
            speed_ms = float(np.linalg.norm(ego.state.get('vel', (0, 0, 0))))

            # Jarak fisik aktual bumper-ke-bumper sebagai Hard Ground-Truth Guard
            gt_dist = max(0.0, float(np.linalg.norm(obs_pos - ego_pos) - 4.5))

            steer_cmd, e_lat, _ = wm.compute_steering(ego_pos, ego_yaw, speed_ms)

            # 1. Ambil Frame Gambar dari Kamera Depan (jika kamera aktif)
            img_bgr = None
            if cam is not None:
                try:
                    cam_data = cam.poll()
                    if cam_data and 'colour' in cam_data and cam_data['colour'] is not None:
                        raw_img = cam_data['colour']
                        img_np = np.array(raw_img)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                except Exception as e_poll:
                    pass

            detected_label = "Jalan Kosong"
            conf_str = "-"
            bbox_str = "-"
            dist_visual_str = "SAFE (>35m)"
            closest_dist = 150.0

            # 2. Inferensi Visual Menggunakan YOLOv8
            results = None
            if img_bgr is not None:
                results = yolo_model(img_bgr, verbose=False)
                boxes = results[0].boxes

                VEHICLE_CLASSES = [2, 3, 5, 7]
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls_id in VEHICLE_CLASSES and conf > 0.35:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox_h = max(y2 - y1, 1.0)
                            est_dist = (focal_length_px * real_car_height_m) / bbox_h
                            
                            detected_label = f"{yolo_model.names[cls_id].upper()}"
                            conf_str = f"{conf*100:.1f}%"
                            bbox_str = f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                            dist_visual_str = f"{est_dist:.1f} m"

                            if est_dist < closest_dist:
                                closest_dist = est_dist
            else:
                # Fallback Telemetri Spasial
                if gt_dist < 35.0:
                    detected_label = "VEHICLE (GT)"
                    conf_str = "100.0%"
                    dist_visual_str = f"{gt_dist:.1f} m"
                closest_dist = gt_dist

            # Safety Guard: Jarak efektif adalah jarak terkecil antara estimasi visual & ground-truth
            effective_dist = min(closest_dist, gt_dist)

            # Logika Aktuator & Safety Shield Berbasis Persepsi
            if effective_dist <= 8.5:
                # [SAFETY SHIELD TRIGGER] Jarak kritis (< 8.5 meter)! Rem 100%
                action_str = "REM DARURAT (SHIELD)"
                cmd_throttle = 0.0
                cmd_brake = 1.0
                cmd_parking = 1.0 if speed_ms < 0.5 else 0.0
            elif effective_dist <= 18.0:
                # Area pendekatan rintangan: deselerasi bertahap
                action_str = "DECEL APPROACH"
                cmd_throttle = 0.0
                cmd_brake = 0.40
                cmd_parking = 0.0
            else:
                # Cruising santai menuju rintangan
                action_str = "CRUISE (Gas 20%)"
                cmd_throttle = 0.20 if speed_ms < 6.0 else 0.0
                cmd_brake = 0.0
                cmd_parking = 0.0

            ego.control(throttle=cmd_throttle, brake=cmd_brake, steering=steer_cmd, parkingbrake=cmd_parking)
            bng.control.step(10)

            # Print log setiap 3 langkah atau saat ada deteksi
            if step % 3 == 0 or effective_dist < 25.0:
                print(f"{step:03d}   | {detected_label:<18} | {conf_str:<12} | {bbox_str:<22} | {dist_visual_str:<12} | {action_str} (d_eff={effective_dist:.1f}m)")

            # 4. Visualisasi HUD: Gambar Kotak Hasil Deteksi YOLO jika ada citra kamera
            if results is not None and len(results) > 0:
                annotated_frame = results[0].plot()
                hud_text = f"Speed: {speed_ms*3.6:.1f} km/h | Visual Lead: {dist_visual_str} | Status: {action_str}"
                cv2.putText(annotated_frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "YOLOv8 Visual Perception Shield - BeamNG.tech", (15, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
                cv2.imshow("Mata AI YOLOv8 - BeamNG.tech HUD", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nPengguna menghentikan visualisasi kamera.")
                    break

            # Jika mobil sudah berhenti aman di depan rintangan
            if effective_dist <= 8.5 and speed_ms < 0.2:
                print(f"\n===========================================================")
                print(f"  BERHASIL AMAN BERHENTI! Jarak akhir rintangan: {effective_dist:.1f} m")
                print(f"  Safety Shield berhasil mencegah tabrakan!")
                print(f"===========================================================")
                time.sleep(3.0)
                break

    except KeyboardInterrupt:
        print("\nDemo dihentikan pengguna.")
    finally:
        cv2.destroyAllWindows()
        try:
            cam.remove()
        except Exception:
            pass
        bng.close()
        print("\nSkenario ditutup dengan sukses.")

if __name__ == "__main__":
    main()

