import os
import glob
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from beamng_suburban_env import BeamNGSuburbanEnv

LOG_DIR = "./tensorboard_logs/"
MODEL_DIR = "./models_saved/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_MODEL_PATH = "models_saved/PPO_Longitudinal1D_20261031_011257_final.zip"

class SuburbanMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SuburbanMetricsCallback, self).__init__(verbose)
        self.episode_interventions = 0
        self.min_dist_in_episode = 150.0
        self.speeds_in_episode = []
        self.lateral_errors = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) > 0:
            info = infos[0]
            if info.get("shield_triggered", False):
                self.episode_interventions += 1
            
            curr_dist = info.get("dist_actual", 150.0)
            if curr_dist < self.min_dist_in_episode:
                self.min_dist_in_episode = curr_dist
                
            e_lat = info.get("e_lat", 0.0)
            self.lateral_errors.append(abs(e_lat))

            obs = self.locals.get("new_obs", None)
            if obs is not None:
                self.speeds_in_episode.append(obs[0][0] * 3.6)

        dones = self.locals.get("dones", [False])
        if dones[0]:
            avg_speed = float(np.mean(self.speeds_in_episode)) if self.speeds_in_episode else 0.0
            avg_elat = float(np.mean(self.lateral_errors)) if self.lateral_errors else 0.0
            
            self.logger.record("safety/shield_interventions", self.episode_interventions)
            self.logger.record("safety/min_clearance_m", self.min_dist_in_episode)
            self.logger.record("performance/avg_speed_kmh", avg_speed)
            self.logger.record("lateral/mean_e_lat_m", avg_elat)
            
            self.episode_interventions = 0
            self.min_dist_in_episode = 150.0
            self.speeds_in_episode = []
            self.lateral_errors = []

        return True

def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning PPO pada Jalur Sub-Urban / Mountain")
    parser.add_argument("--route", type=str, default="suburban", choices=["suburban", "mountain_castle", "mountain_village"])
    parser.add_argument("--timesteps", type=int, default=5000, help="Jumlah total langkah pelatihan (default: 5000)")
    args, _ = parser.parse_known_args()

    run_name = f"PPO_{args.route.capitalize()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("===========================================================")
    print(f"      PELATIHAN TRANSFER LEARNING PPO: {run_name}          ")
    print("===========================================================")

    # 1. Inisialisasi Lingkungan
    env = BeamNGSuburbanEnv(route_type=args.route, start_offset_m=45.0, obstacle_distance_m=90.0)

    # 2. Muat atau Buat Model PPO
    initial_model = BASE_MODEL_PATH
    existing_suburban = sorted(glob.glob(os.path.join(MODEL_DIR, "PPO_Suburban_*.zip")))
    if existing_suburban:
        initial_model = existing_suburban[-1]

    if os.path.exists(initial_model):
        print(f"-> Melakukan Transfer Learning dari model: {initial_model}")
        model = PPO.load(
            initial_model,
            env=env,
            device="cpu",
            tensorboard_log=LOG_DIR
        )
    else:
        print("-> Membangun model PPO baru...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cpu"
        )

    callback = SuburbanMetricsCallback()

    print(f"\n--- Memulai Pelatihan ({args.timesteps} Timesteps) ---")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=run_name,
            reset_num_timesteps=False
        )
        print("\nPelatihan selesai dengan sukses!")
    except KeyboardInterrupt:
        print("\nPelatihan dihentikan secara manual.")
    finally:
        save_path = os.path.join(MODEL_DIR, f"{run_name}_final")
        model.save(save_path)
        print(f"Model berhasil disimpan ke: {save_path}.zip")
        env.close()

if __name__ == "__main__":
    main()

