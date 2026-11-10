import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from beamng_env import BeamNGSlopeShieldEnv

LOG_DIR = "./tensorboard_logs/"
MODEL_DIR = "./models_saved/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class ShieldMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ShieldMetricsCallback, self).__init__(verbose)
        self.episode_interventions = 0
        self.min_dist_in_episode = 150.0
        self.speeds_in_episode = []
        self.collision_in_episode = 0
        self.safe_stop_in_episode = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) > 0:
            info = infos[0]
            if info.get("shield_triggered", False):
                self.episode_interventions += 1
            
            if info.get("collision", False):
                self.collision_in_episode = 1
                
            if info.get("safe_stop", False):
                self.safe_stop_in_episode = 1

            curr_dist = info.get("dist_actual", 150.0)
            if curr_dist < self.min_dist_in_episode:
                self.min_dist_in_episode = curr_dist
                
            obs = self.locals.get("new_obs", None)
            if obs is not None:
                self.speeds_in_episode.append(obs[0][0] * 3.6)

        dones = self.locals.get("dones", [False])
        if dones[0]:
            avg_speed = float(np.mean(self.speeds_in_episode)) if self.speeds_in_episode else 0.0
            
            self.logger.record("safety/shield_interventions", self.episode_interventions)
            self.logger.record("safety/min_clearance_m", self.min_dist_in_episode)
            self.logger.record("safety/collision", self.collision_in_episode)
            self.logger.record("safety/safe_stop", self.safe_stop_in_episode)
            self.logger.record("performance/avg_speed_kmh", avg_speed)
            
            self.episode_interventions = 0
            self.min_dist_in_episode = 150.0
            self.speeds_in_episode = []
            self.collision_in_episode = 0
            self.safe_stop_in_episode = 0

        return True

def main():
    print("Menginisialisasi Lingkungan BeamNG (Longitudinal 1D)...")
    env = BeamNGSlopeShieldEnv()

    run_name = f"PPO_Longitudinal1D_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Inisialisasi PPO pada CPU untuk eliminasi latency transfer GPU pada MLP 3-fitur
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

    metrics_callback = ShieldMetricsCallback()

    print(f"\n--- Memulai Pelatihan Model: {run_name} ---")
    TOTAL_TIMESTEPS = 3000
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=metrics_callback,
            tb_log_name=run_name,
            reset_num_timesteps=False
        )
        print("\nPelatihan selesai.")
    except KeyboardInterrupt:
        print("\nPelatihan dihentikan secara manual.")
    finally:
        save_path = os.path.join(MODEL_DIR, f"{run_name}_final")
        model.save(save_path)
        print(f"Model berhasil disimpan ke: {save_path}.zip")
        env.close()

if __name__ == "__main__":
    main()
