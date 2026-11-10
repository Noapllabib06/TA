import os
import glob
import time
import argparse
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from beamng_suburban_env import BeamNGSuburbanEnv

MODEL_DIR = "./models_saved/"

def get_latest_model():
    models = sorted(glob.glob(os.path.join(MODEL_DIR, "PPO_Suburban_Solo_*_final.zip")))
    if not models:
        raise FileNotFoundError("Tidak ditemukan berkas model final di folder models_saved/")
    return models[-1]

def evaluate(episodes=5, model_path=None, start_offset_m=48.0, speed_kmh=80.0):
    if model_path is None:
        model_path = get_latest_model()

    print("=" * 70)
    print("   UJI DETERMINISTIK (EVALUASI MURNI BEBAS NOISE EKSPLORASI)")
    print(f"   Model       : {model_path}")
    print(f"   Start Offset: {start_offset_m} m | Target Kecepatan: {speed_kmh} km/h")
    print(f"   Episodes    : {episodes}")
    print("=" * 70)

    env = BeamNGSuburbanEnv(
        route_type="suburban",
        start_offset_m=start_offset_m,
        has_obstacle=False,
        target_cruise_kmh=speed_kmh,
        use_yolo=True
    )
    env.max_steps = 2500

    print(f"-> Memuat bobot agen dari: {model_path} ...")
    model = PPO.load(model_path, env=env, device="cpu")

    results = []
    for ep in range(1, episodes + 1):
        t_start = time.time()
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        done = False
        steps = 0
        cbe_count = 0
        cls_count = 0
        info = {}

        while not done:
            # deterministic=True mematikan seluruh noise eksplorasi Gaussian PPO
            action, _ = model.predict(obs, deterministic=True)
            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                obs, reward, done, info = step_ret
            steps += 1

            if info.get("cbe_intervened", False):
                cbe_count += 1
            if info.get("cls_intervened", False):
                cls_count += 1

        ep_dur = max(0.1, time.time() - t_start)
        t_now = datetime.now().strftime("%H:%M:%S")
        prog = info.get("progress_m", 0.0)
        status = info.get("termination_reason", "SELESAI")
        shd_pct = ((cbe_count + cls_count) / max(steps, 1)) * 100.0
        elat_avg = info.get("e_lat", 0.0)

        print(f"[{t_now}] [TEST EP {ep:02d}] Jarak: {prog:5.1f}m | Steps: {steps:3d} ({ep_dur:4.1f}s) | Shield: {shd_pct:4.1f}% | e_lat: {elat_avg:+.2f}m | Status: {status}")
        results.append((prog, steps, shd_pct))

    env.close()

    progs = [r[0] for r in results]
    steps_all = [r[1] for r in results]
    shd_all = [r[2] for r in results]

    print("\n" + "=" * 70)
    print(f"HASIL EVALUASI DETERMINISTIK ({episodes} Episode):")
    print(f" * Rata-rata Jarak Jelajah : {np.mean(progs):.1f} m (Maksimum: {np.max(progs):.1f} m, Minimum: {np.min(progs):.1f} m)")
    print(f" * Rata-rata Langkah/Episode: {np.mean(steps_all):.1f} steps")
    print(f" * Rata-rata Intervensi    : {np.mean(shd_all):.1f}%")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Uji Deterministik PPO Suburban Solo (Bebas Noise)")
    parser.add_argument("--episodes", type=int, default=5, help="Jumlah episode evaluasi (default: 5)")
    parser.add_argument("--model_path", type=str, default=None, help="Path ke model .zip (default: model final terbaru)")
    parser.add_argument("--start_offset", type=float, default=48.0, help="Offset awal rute dalam meter (default: 48.0)")
    parser.add_argument("--speed", type=float, default=80.0, help="Target cruise speed km/h (default: 80.0)")
    args = parser.parse_args()

    evaluate(
        episodes=args.episodes,
        model_path=args.model_path,
        start_offset_m=args.start_offset,
        speed_kmh=args.speed
    )

if __name__ == "__main__":
    main()

