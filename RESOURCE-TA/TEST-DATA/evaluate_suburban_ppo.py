import os
import sys
import time
import argparse
import numpy as np
import cv2
from stable_baselines3 import PPO

from beamng_suburban_env import BeamNGSuburbanEnv

MODEL_PATH = "models_saved/PPO_Longitudinal1D_20261031_011257_final.zip"

def main():
    parser = argparse.ArgumentParser(description="Evaluasi Model PPO Otonom + Safety Shield pada Jalur Sub-Urban")
    parser.add_argument("--route", type=str, default="suburban", choices=["suburban", "mountain_castle", "mountain_village"],
                        help="Pilihan rute: suburban, mountain_castle, atau mountain_village")
    parser.add_argument("--steps", type=int, default=200,
                        help="Jumlah langkah preview (default: 200 step ~ 600 meter)")
    parser.add_argument("--obs_dist", type=float, default=None,
                        help="Jarak awal mobil rintangan di depan (meter, default: 65m jika moving, 110m jika static)")
    parser.add_argument("--moving", action="store_true",
                        help="Aktifkan mobil rintangan bergerak maju (ACC scenario)")
    parser.add_argument("--solo", action="store_true",
                        help="Jelajah rute sendirian tanpa kendaraan rintangan (Solo Autonomous Cruising)")
    parser.add_argument("--full", action="store_true",
                        help="Jelajah satu rute penuh dari garis start hingga finish (~3.55 km / ~1600 steps)")
    parser.add_argument("--speed", type=float, default=80.0,
                        help="Target kecepatan jelajah di jalan lurus (km/jam, default: 80.0)")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path ke checkpoint model PPO .zip")
    parser.add_argument("--start_offset", type=float, default=48.0,
                        help="Titik awal spawn mobil sepanjang lintasan (meter, default: 48.0 = jalan utama luar SPBU)")
    parser.add_argument("--no_yolo", dest="yolo", action="store_false",
                        help="Nonaktifkan modul persepsi visual YOLOv8-seg")
    parser.set_defaults(yolo=True)
    args, _ = parser.parse_known_args()

    if args.full:
        args.steps = 1600

    # Tentukan mode evaluasi
    if args.solo:
        mode_header = f"SOLO AUTONOMOUS CRUISING (Target: {args.speed:.0f} km/jam) {'[SATU RUTE PENUH 3.55 KM]' if args.full else ''}"
        mode_obs_str = f"TIDAK ADA RINTANGAN (Jalan Kosong, Cruise Max {args.speed:.0f} km/h)"
        target_obs_dist = 999.0
    elif args.moving:
        mode_header = "MOVING OBSTACLE (ACC Adaptive Cruise Control)"
        mode_obs_str = "BERGERAK MAJU (ACC Traffic ~20 km/jam)"
        target_obs_dist = args.obs_dist if args.obs_dist is not None else 65.0
    else:
        mode_header = "STATIC OBSTACLE (Collision Avoidance Shield)"
        mode_obs_str = "STATIS TERKUNCI (Rem Parkir)"
        target_obs_dist = args.obs_dist if args.obs_dist is not None else 110.0

    print("===========================================================")
    print(f"      EVALUASI PREVIEW MODEL DRL PPO: {mode_header}")
    print("===========================================================")
    
    # Jika model default digunakan, periksa apakah ada model khusus Suburban yang lebih baru
    target_model_file = args.model
    if target_model_file == MODEL_PATH and os.path.exists("models_saved"):
        sub_models = [
            os.path.join("models_saved", f) for f in os.listdir("models_saved")
            if f.startswith("PPO_Suburban") and f.endswith(".zip")
        ]
        if sub_models:
            sub_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            target_model_file = sub_models[0]
            print(f"-> Terdeteksi model khusus Sub-Urban terbaru: {target_model_file}")

    if not os.path.exists(target_model_file):
        print(f"Error: Model {target_model_file} tidak ditemukan!")
        return

    # 1. Muat Model PPO pada CPU
    print(f"-> Memuat Otak AI (PPO) dari: {target_model_file}")
    model = PPO.load(target_model_file, device="cpu")
    print("-> Model AI berhasil dimuat pada CPU!")

    # 2. Inisialisasi Lingkungan Gymnasium Terintegrasi
    print(f"-> Menginisialisasi Lingkungan '{args.route}' ({args.steps} Steps, Speed {args.speed:.0f} km/h)...")
    print(f"-> Mode Mobil: {mode_obs_str}")
    
    env = BeamNGSuburbanEnv(
        route_type=args.route,
        start_offset_m=args.start_offset,
        obstacle_distance_m=target_obs_dist,
        is_moving_obstacle=args.moving,
        has_obstacle=(not args.solo),
        target_cruise_kmh=args.speed,
        use_yolo=args.yolo
    )
    env.max_steps = args.steps

    obs, info = env.reset()

    has_gui = False
    if args.yolo:
        try:
            cv2.namedWindow("Mata AI: Live YOLOv8-seg Perception HUD", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mata AI: Live YOLOv8-seg Perception HUD", 720, 540)
            # Segera render frame awal atau placeholder agar Windows message queue aktif (tidak 'Not Responding')
            initial_hud = getattr(env, 'last_yolo_hud_frame', None)
            if initial_hud is not None:
                cv2.imshow("Mata AI: Live YOLOv8-seg Perception HUD", initial_hud)
            else:
                splash = np.zeros((540, 720, 3), dtype=np.uint8)
                cv2.putText(splash, "Menghubungkan Kamera Bemper & YOLOv8-seg...", (40, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                cv2.imshow("Mata AI: Live YOLOv8-seg Perception HUD", splash)
            cv2.waitKey(1)
            print("-> Jendela Visual OpenCV HUD BERHASIL diaktifkan!")
            print("   (Jendela menampilkan live video kamera bemper, poligon YOLOv8-seg, dan PiP Raw Camera)")
            has_gui = True
        except (cv2.error, Exception) as e:
            has_gui = False
            print(f"\n[CATATAN] Jendela GUI OpenCV gagal dibuka: {e}")
            print("         Inferensi visual YOLOv8-seg tetap berjalan 100% penuh di background!")

    print("\n--- Memulai Evaluasi Otonom (DRL PPO + Stanley + SA-TTC + Kinematic Profiler) ---")
    dist_header = "Steer / Curvature" if args.solo else "Dist Lead"
    print(f"{'Step':<5} | {'Speed':<10} | {'V_target':<10} | {'Gas / Rem':<15} | {dist_header:<22} | {'e_lat (CG / Curb)':<20} | {'Status'}")
    print("-" * 115)

    total_reward = 0.0
    safe_stop_count = 0

    try:
        for step in range(args.steps):
            # Otak PPO memprediksi aksi gas/rem dari observasi [kecepatan, jarak lidar, pitch]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            speed_kmh = obs[0] * 3.6
            v_target_kmh = info.get("v_target", 10.0) * 3.6
            dist_lidar = obs[1]
            curv_val = info.get("curv", 0.0)
            thr = info.get("throttle", 0.0)
            brk = info.get("brake", 0.0)
            steer_val = info.get("steering", 0.0)

            pedal_str = f"T:{thr:4.2f} B:{brk:4.2f}"

            if info.get("cbe_intervened", False):
                status = "POLISI 1: CBE CURB"
            elif info.get("cls_intervened", False):
                status = "POLISI 2: CLS STABIL"
            elif info["shield_triggered"]:
                if args.solo:
                    if brk > 0.05:
                        status = "DECEL TIKUNGAN"
                    else:
                        status = "SPEED GOVERNOR"
                else:
                    status = "SHIELD JARAK AMAN"
            elif info["safe_stop"]:
                status = "AMAN STOP"
            elif args.moving and not args.solo and dist_lidar < 28.0 and speed_kmh > 5.0:
                status = "ACC CAR-FOLLOWING"
            elif args.solo:
                if abs(curv_val) > 0.06:
                    status = "CORNERING KANAN" if curv_val < 0 else "CORNERING KIRI"
                elif speed_kmh >= 65.0:
                    status = "HIGH-SPEED CRUISE"
                elif thr > 0.4:
                    status = "ACCEL TO TARGET"
                else:
                    status = "STEADY CRUISE"
            else:
                status = "AI CRUISING"

            prog_m = info.get("progress_m", 0.0)
            tot_m = info.get("total_route_m", 3157.0)
            pct = (prog_m / max(tot_m, 1.0)) * 100.0

            ref_off = info.get("ref_offset", 0.0)
            mode_ref = "CoM" if abs(ref_off) < 0.1 else ("CoRA" if abs(ref_off) > 1.1 else f"R{abs(ref_off):.1f}")

            if args.solo:
                col_info = f"Str:{steer_val:+.2f} ({curv_val:+.3f}|{mode_ref})"
            else:
                col_info = f"{dist_lidar:5.1f} m (d_s:{info['d_safe']:3.1f})"

            elat_cg = info['e_lat']
            d_curb = info.get('d_curb_eff', 4.0)
            d_phys = info.get('d_curb_phys', 4.0)
            d_yolo = info.get('d_curb_yolo', 4.0)
            if d_phys < 3.8 and d_yolo < 3.8:
                curb_disp = f"Y:{d_yolo:3.1f}|P:{d_phys:3.1f}"
            elif d_phys < 3.8:
                curb_disp = f"P:{d_phys:3.1f}m"
            elif d_yolo < 3.8:
                curb_disp = f"Y:{d_yolo:3.1f}m"
            else:
                curb_disp = "SAFE"
            elat_str = f"{elat_cg:+.2f}m(C:{curb_disp})"

            print(f"{step+1:04d}  | {speed_kmh:5.1f} km/h | {v_target_kmh:5.1f} km/h | {pedal_str:<15} | {col_info:<22} | {elat_str:<18} | {status:<16} [{prog_m:6.1f}m / {tot_m:.0f}m ({pct:4.1f}%)]")

            # Tampilkan Jendela Live Visual HUD sebelum evaluasi break agar frame saat kejadian tetap ter-render
            if args.yolo and has_gui:
                try:
                    yolo_hud = info.get("yolo_hud_frame", None)
                    if yolo_hud is not None:
                        cv2.imshow("Mata AI: Live YOLOv8-seg Perception HUD", yolo_hud)
                    else:
                        # Fallback Telemetry HUD jika kamera BeamNG masih sinkronisasi frame awal
                        fallback_hud = np.zeros((540, 720, 3), dtype=np.uint8)
                        cv2.rectangle(fallback_hud, (20, 20), (700, 520), (25, 25, 25), -1)
                        cv2.rectangle(fallback_hud, (20, 20), (700, 520), (0, 255, 200), 2)
                        cv2.putText(fallback_hud, "MATA AI: TELEMETRI SENSOR KENDARAAN OTONOM", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
                        cv2.putText(fallback_hud, f"Kecepatan: {speed_kmh:.1f} km/jam | Target: {v_target_kmh:.1f} km/jam", (40, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.putText(fallback_hud, f"Gas: {thr:.2f} | Rem: {brk:.2f} | Kemudi: {steer_val:+.3f}", (40, 145),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.putText(fallback_hud, f"Cross-Track Error e_lat: {elat_cg:+.3f} meter", (40, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
                        cv2.putText(fallback_hud, f"Jarak Trotoar Efektif  : {d_curb:.2f} meter", (40, 215),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
                        cv2.putText(fallback_hud, f"Status: {status}", (40, 260),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        cv2.putText(fallback_hud, "Sinkronisasi kamera bemper BeamNG...", (40, 480),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                        cv2.imshow("Mata AI: Live YOLOv8-seg Perception HUD", fallback_hud)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n-> Visualisasi HUD ditutup oleh pengguna (simulasi tetap berjalan).")
                        has_gui = False
                        cv2.destroyAllWindows()
                except Exception:
                    pass

            term_reason = info.get("termination_reason", "")
            if info.get("route_completed", False):
                print("\n" + "=" * 80)
                print(f"  LUAR BIASA! MOBIL BERHASIL MENUNTASKAN SATU RUTE PENUH ({tot_m/1000.0:.2f} KM)!")
                print(f"  Total Jarak Tempuh : {prog_m:.1f} meter")
                print("  Status Tabrakan    : 0 TABRAKAN (100% Zero-Collision)")
                print("  Status Jalur       : BERTAHAN DI DALAM LAJUR HINGGA GARIS FINISH")
                print("=" * 80)
                break

            if info.get("curb_climb", False):
                print(f"\n-> [PENALTI -35.0] Pelanggaran Fisik: Roda melindas trotoar di Step {step+1:04d} (Roll={info.get('roll_deg', 0.0):+.1f}°, e_lat={info['e_lat']:+.2f} m)! Episode di-reset.")
                break

            if info.get("off_track", False):
                print(f"\n-> [PENALTI -25.0] Keluar Jalur (Off-Track): Mobil keluar dari badan jalan di Step {step+1:04d} (e_lat={info['e_lat']:+.2f} m > 2.0m)!")
                break

            if info.get("collision_obstacle", False):
                print(f"\n-> [PENALTI -60.0] Tabrakan Fatal: Menabrak mobil rintangan di Step {step+1:04d} (Jarak={dist_lidar:.2f} m)!")
                break

            if info.get("collision_structural", False):
                print(f"\n-> [PENALTI -45.0] Benturan Keras: Deformasi struktur sasis menabrak tiang/dinding di Step {step+1:04d}!")
                break

            if info.get("stuck", False):
                print(f"\n-> [PENALTI -15.0] Mobil Tersangkut: Terhenti terhadap trotoar/rintangan selama >3 step!")
                break

            if info["collision"]:
                print(f"\n-> Terjadi kontak tabrakan di Step {step+1:04d} (e_lat={info['e_lat']:+.2f} m)!")
                break

            # Jika rintangan statis dan mobil berhenti aman sempurna
            if info["safe_stop"] and not args.moving and not args.solo:
                safe_stop_count += 1
                if safe_stop_count >= 15:
                    print(f"\n-> Berhenti sempurna di depan rintangan! Menahan posisi aman selama 15 step.")
                    print(f"-> HASIL AKHIR: LULUS UJI KESELAMATAN (Jarak aman: {dist_lidar:.2f} m).")
                    break

            if terminated and not info["collision"] and not info["safe_stop"] and not info.get("route_completed", False):
                print(f"\n-> Terminasi dini di Step {step+1:04d}: {term_reason}")
                break

            time.sleep(0.01)

    finally:
        if args.yolo and has_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        env.close()
    print("=" * 80)
    print(f"Evaluasi Selesai ({args.steps} Steps). Total Episode Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
