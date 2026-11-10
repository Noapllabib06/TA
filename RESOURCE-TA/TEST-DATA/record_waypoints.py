import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from beamngpy import BeamNGpy, Scenario, Vehicle

BNG_HOME = r"C:\BeamNG\BeamNG.tech.v0.39.4.0"

# Konfigurasi preset titik awal jalan untuk Sub-Urban & Mountain
MAP_PRESETS = {
    "1": {
        "name": "Mountain (Italy Village)",
        "map": "italy",
        "scenario_name": "record_italy_mountain",
        "pos": (262.39, -891.67, 246.92),
        "rot": (0.0, 0.0, -0.2588, 0.9659),
        "output_file": "waypoints_italy_mountain.csv"
    },
    "2": {
        "name": "Sub-Urban (East Coast Town)",
        "map": "east_coast_usa",
        "scenario_name": "record_suburban_eastcoast",
        "pos": (723.16, -5.94, 52.08),
        "rot": (0.0, 0.0, 0.7071, 0.7071),
        "output_file": "waypoints_suburban_town.csv"
    },
    "3": {
        "name": "Extreme Mountain (Italy Castle Pass)",
        "map": "italy",
        "scenario_name": "record_italy_castle",
        "pos": (-969.64, 953.63, 392.48),
        "rot": (0.0, 0.0, 0.0, 1.0),
        "output_file": "waypoints_italy_castle.csv"
    }
}

def select_preset(preset_arg=None):
    if preset_arg and str(preset_arg) in MAP_PRESETS:
        return MAP_PRESETS[str(preset_arg)]

    print("\n===========================================================")
    print("      PEREKAM WAYPOINTS JALUR OTONOM (BeamNG.tech)         ")
    print("===========================================================")
    print("Pilih medan yang ingin Anda rekam:")
    print("  [1] Mountain Terrain   - Italy Mountain Village (Elevasi Curam & Berliku)")
    print("  [2] Sub-Urban Terrain  - East Coast USA Town (Jalan Lingkungan & Tikungan)")
    print("  [3] Extreme Mountain   - Italy Castle Pass (Elevasi Puncak 392m)")
    
    choice = input("\nMasukkan pilihan [1/2/3] (Default: 2): ").strip()
    if choice not in MAP_PRESETS:
        choice = "2"
    return MAP_PRESETS[choice]

def main():
    parser = argparse.ArgumentParser(description="Perekam Waypoints Jalur Otonom BeamNG.tech")
    parser.add_argument("--preset", type=str, default="2", choices=["1", "2", "3"],
                        help="Pilihan preset peta: 1 (Mountain), 2 (Suburban), 3 (Castle). Default: 2")
    parser.add_argument("--out", type=str, default="",
                        help="Nama file CSV output kustom. Kosongkan untuk menggunakan default preset.")
    args, _ = parser.parse_known_args()

    preset = select_preset(args.preset)
    map_id = preset["map"]
    sc_name = preset["scenario_name"]
    start_pos = preset["pos"]
    start_rot = preset["rot"]
    out_csv = args.out if args.out else preset["output_file"]

    print(f"\n-> Memilih Preset: {preset['name']}")
    print(f"-> Membuka BeamNG.tech pada peta '{map_id}'...")
    
    bng = BeamNGpy('localhost', 64256, home=BNG_HOME)
    bng.open(launch=True)

    scenario = Scenario(map_id, sc_name)
    ego = Vehicle('ego_vehicle', model='etk800')
    scenario.add_vehicle(ego, pos=start_pos, rot_quat=start_rot)
    
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()

    # Transmisi arcade otomatis agar langsung jalan dengan gas/rem
    ego.set_shift_mode('arcade')

    print("\n" + "=" * 65)
    print("               SIMULATOR SIAP! INSTRUKSI MENYETIR:         ")
    print("=" * 65)
    print("1. Klik jendela game BeamNG.tech untuk memfokuskan kendali.")
    print("2. Setir mobil secara santai (gunakan tombol Panah / WASD).")
    print("3. Jarak yang disarankan: 300 s.d. 500 METER (sekitar 30 - 60 detik).")
    print("4. Setelah rute yang Anda inginkan selesai, tekan Ctrl + C di terminal ini.")
    print("=" * 65 + "\n")

    waypoints = []
    total_dist = 0.0
    last_pos = None
    step_idx = 0

    try:
        while True:
            ego.sensors.poll()
            pos = np.array(ego.state.get('pos', (0, 0, 0)))
            dir_vec = np.array(ego.state.get('dir', (1, 0, 0)))
            vel_vec = np.array(ego.state.get('vel', (0, 0, 0)))

            speed_kmh = float(np.linalg.norm(vel_vec) * 3.6)
            pitch_rad = float(np.arcsin(np.clip(dir_vec[2], -1.0, 1.0)))
            pitch_deg = float(np.degrees(pitch_rad))

            if last_pos is None:
                last_pos = pos
                waypoints.append({
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "dir_x": dir_vec[0], "dir_y": dir_vec[1], "dir_z": dir_vec[2],
                    "pitch_rad": pitch_rad, "pitch_deg": pitch_deg,
                    "speed_kmh": speed_kmh, "distance_m": 0.0
                })
            else:
                delta_d = float(np.linalg.norm(pos[:2] - last_pos[:2]))
                # Rekam setiap mobil berpindah sejauh minimal 1.0 meter
                if delta_d >= 1.0:
                    total_dist += delta_d
                    last_pos = pos
                    waypoints.append({
                        "x": pos[0], "y": pos[1], "z": pos[2],
                        "dir_x": dir_vec[0], "dir_y": dir_vec[1], "dir_z": dir_vec[2],
                        "pitch_rad": pitch_rad, "pitch_deg": pitch_deg,
                        "speed_kmh": speed_kmh, "distance_m": total_dist
                    })

                    # Tampilkan status lereng
                    if pitch_deg > 2.0:
                        slope_str = f"Tanjakan (+{pitch_deg:.1f}°)"
                    elif pitch_deg < -2.0:
                        slope_str = f"Turunan ({pitch_deg:.1f}°)"
                    else:
                        slope_str = f"Datar ({pitch_deg:.1f}°)"

                    print(f"\r[REC] Jarak: {total_dist:6.1f} m | Titik: {len(waypoints):4d} | Speed: {speed_kmh:5.1f} km/h | Kontur: {slope_str:<18} | Z: {pos[2]:6.1f} m", end="", flush=True)

            bng.control.step(5)
            time.sleep(0.02)
            step_idx += 1

    except KeyboardInterrupt:
        print("\n\n-> Perekaman dihentikan oleh pengguna.")

    finally:
        if len(waypoints) > 5:
            # Auto-backup file lama jika sudah ada agar data sebelumnya tetap aman
            if os.path.exists(out_csv):
                import shutil
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = out_csv.replace(".csv", f"_backup_{ts}.csv")
                shutil.copy2(out_csv, backup_name)
                print(f"-> Backup file lama tersimpan ke: {backup_name}")

            df = pd.DataFrame(waypoints)
            df.to_csv(out_csv, index=False)
            print("\n" + "=" * 65)
            print("                 RINGKASAN LINTASAN TERCATAT               ")
            print("=" * 65)
            print(f"File Tersimpan        : {os.path.abspath(out_csv)}")
            print(f"Total Jarak Rute      : {total_dist:.2f} meter")
            print(f"Jumlah Titik Waypoint : {len(waypoints)} titik")
            print(f"Elevasi Awal          : {df['z'].iloc[0]:.2f} m")
            print(f"Elevasi Akhir         : {df['z'].iloc[-1]:.2f} m")
            print(f"Perubahan Elevasi (dZ): {df['z'].iloc[-1] - df['z'].iloc[0]:+.2f} m")
            print(f"Tanjakan Maksimum     : {df['pitch_deg'].max():+.2f}°")
            print(f"Turunan Maksimum      : {df['pitch_deg'].min():+.2f}°")
            print("=" * 65)
            print("\n-> Lintasan berhasil direkam dan siap digunakan untuk pelatihan lateral!")
        else:
            print("\nTitik waypoint terlalu sedikit (< 5 titik). Perekaman dibatalkan.")

        bng.close()

if __name__ == "__main__":
    main()

