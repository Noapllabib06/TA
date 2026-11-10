import os
import glob
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from beamng_suburban_env import BeamNGSuburbanEnv

LOG_DIR = "./tensorboard_logs/"
MODEL_DIR = "./models_saved/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_MODEL_PATH = "models_saved/PPO_Longitudinal1D_20261031_011257_final.zip"

class SoloMetricsCallback(BaseCallback):
    """
    Callback untuk merekam metrik kelayakan proses pelatihan (Training Feasibility):
    1. Feasibility / Autonomy Index:
       - Shield Intervention Ratio (% langkah di mana CBE/CLS mengambil alih kendali)
       - Rekor dan rata-rata jarak jelajah rute (progress_m max & mean)
    2. Lateral Tracking Precision:
       - Mean e_lat, Max e_lat, Max rear-right e_lat (anti-cut-in)
    3. Curb Clearance & Safety Invariant:
       - Min effective curb distance (harus > 0.0m)
       - Total collision, off-track, curb-climb, stuck events
    4. Speed Tracking Quality:
       - Average speed (km/jam)
       - Speed tracking error (|v - v_target| km/jam)
    """
    def __init__(self, verbose=1, start_ep=1):
        super(SoloMetricsCallback, self).__init__(verbose)
        self.ep_count = start_ep - 1
        self.rollout_count = 0

        # Episode-level buffers
        self.speeds_in_episode = []
        self.lateral_errors = []
        self.rear_right_errors = []
        self.curb_distances = []
        self.speed_errors = []
        self.cbe_interventions = 0
        self.cls_interventions = 0
        self.ep_steps = 0

        # Rollout-level aggregate buffers (merangkum seluruh langkah & episode dalam 1 rollout buffer)
        self.rollout_speeds = []
        self.rollout_lateral_errors = []
        self.rollout_rear_right_errors = []
        self.rollout_curb_distances = []
        self.rollout_speed_errors = []
        self.rollout_progress_m = []
        self.rollout_total_steps = 0
        self.rollout_shield_steps = 0
        self.rollout_cbe_steps = 0
        self.rollout_cls_steps = 0
        self.rollout_collision_count = 0
        self.rollout_off_track_count = 0
        self.rollout_curb_climb_count = 0
        self.rollout_stuck_count = 0

    def _on_step(self) -> bool:
        self.ep_steps += 1
        self.rollout_total_steps += 1
        infos = self.locals.get("infos", [])
        last_info = {}
        if len(infos) > 0:
            last_info = infos[0]
            e_lat = last_info.get("e_lat", 0.0)
            self.lateral_errors.append(abs(e_lat))
            self.rollout_lateral_errors.append(abs(e_lat))

            e_rr = last_info.get("e_lat_rear_right", e_lat)
            self.rear_right_errors.append(abs(e_rr))
            self.rollout_rear_right_errors.append(abs(e_rr))

            d_curb = last_info.get("d_curb_eff", last_info.get("d_curb_phys", 4.0))
            self.curb_distances.append(d_curb)
            self.rollout_curb_distances.append(d_curb)

            cbe_act = last_info.get("cbe_intervened", False)
            cls_act = last_info.get("cls_intervened", False)
            if cbe_act:
                self.cbe_interventions += 1
                self.rollout_cbe_steps += 1
            if cls_act:
                self.cls_interventions += 1
                self.rollout_cls_steps += 1
            if cbe_act or cls_act:
                self.rollout_shield_steps += 1

            obs = self.locals.get("new_obs", None)
            if obs is not None:
                cur_spd_kmh = obs[0][0] * 3.6
                self.speeds_in_episode.append(cur_spd_kmh)
                self.rollout_speeds.append(cur_spd_kmh)

                v_tgt_kmh = last_info.get("v_target", cur_spd_kmh / 3.6) * 3.6
                spd_err = abs(cur_spd_kmh - v_tgt_kmh)
                self.speed_errors.append(spd_err)
                self.rollout_speed_errors.append(spd_err)

        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.ep_count += 1
            avg_speed = float(np.mean(self.speeds_in_episode)) if self.speeds_in_episode else 0.0
            avg_elat = float(np.mean(self.lateral_errors)) if self.lateral_errors else 0.0
            max_elat = float(np.max(self.lateral_errors)) if self.lateral_errors else 0.0
            max_rr_elat = float(np.max(self.rear_right_errors)) if self.rear_right_errors else 0.0
            min_curb = float(np.min(self.curb_distances)) if self.curb_distances else 4.0

            curb_climb = last_info.get("curb_climb", False)
            off_track = last_info.get("off_track", False)
            collision = last_info.get("collision", False)
            stuck = last_info.get("stuck", False)
            progress_m = last_info.get("progress_m", 0.0)
            total_m = max(last_info.get("total_route_m", 3158.0), 1.0)
            pct = (progress_m / total_m) * 100.0
            term_reason = last_info.get("termination_reason", "SELESAI")

            if collision:
                self.rollout_collision_count += 1
            if off_track:
                self.rollout_off_track_count += 1
            if curb_climb:
                self.rollout_curb_climb_count += 1
            if stuck:
                self.rollout_stuck_count += 1
            self.rollout_progress_m.append(progress_m)

            status_str = f"STATUS: {term_reason}"
            curb_str = f"{min_curb:.2f}m" if min_curb < 3.8 else "AMAN"
            shield_pct = ((self.cbe_interventions + self.cls_interventions) / max(1, self.ep_steps)) * 100.0
            shield_str = f"CBE:{self.cbe_interventions}|CLS:{self.cls_interventions} ({shield_pct:.0f}%)"
            print(f"-> [EP {self.ep_count:03d} | Total Step {self.num_timesteps}] Steps: {self.ep_steps:3d} | Spd: {avg_speed:4.1f} km/h | e_lat: {avg_elat:+.2f}m (Max: {max_elat:.2f}m, RR: {max_rr_elat:.2f}m) | Min Curb: {curb_str} | Shield: {shield_str} | Progres: {progress_m:5.1f}m ({pct:4.1f}%) | {status_str}")

            self.speeds_in_episode = []
            self.lateral_errors = []
            self.rear_right_errors = []
            self.curb_distances = []
            self.speed_errors = []
            self.cbe_interventions = 0
            self.cls_interventions = 0
            self.ep_steps = 0

        return True

    def _on_rollout_end(self) -> None:
        """
        Dipanggil setiap kali 1 rollout buffer penuh (tiap n_steps langkah).
        Merekam metrik evaluasi kelayakan komprehensif ke SB3 logger (dicetak di tabel ASCII).
        """
        self.rollout_count += 1

        total_steps = max(1, self.rollout_total_steps)
        shield_ratio_pct = (self.rollout_shield_steps / total_steps) * 100.0
        cbe_ratio_pct = (self.rollout_cbe_steps / total_steps) * 100.0
        cls_ratio_pct = (self.rollout_cls_steps / total_steps) * 100.0

        avg_spd = float(np.mean(self.rollout_speeds)) if self.rollout_speeds else 0.0
        spd_err = float(np.mean(self.rollout_speed_errors)) if self.rollout_speed_errors else 0.0
        mean_elat = float(np.mean(self.rollout_lateral_errors)) if self.rollout_lateral_errors else 0.0
        max_elat = float(np.max(self.rollout_lateral_errors)) if self.rollout_lateral_errors else 0.0
        max_rr_elat = float(np.max(self.rollout_rear_right_errors)) if self.rollout_rear_right_errors else 0.0
        min_curb = float(np.min(self.rollout_curb_distances)) if self.rollout_curb_distances else 4.0
        mean_curb = float(np.mean(self.rollout_curb_distances)) if self.rollout_curb_distances else 4.0

        max_prog = float(np.max(self.rollout_progress_m)) if self.rollout_progress_m else 0.0
        mean_prog = float(np.mean(self.rollout_progress_m)) if self.rollout_progress_m else 0.0

        # Kategori 1: Kelayakan Otonomi & Progres (Feasibility)
        self.logger.record("feasibility/shield_ratio_pct", shield_ratio_pct)
        self.logger.record("feasibility/max_progress_m", max_prog)
        self.logger.record("feasibility/mean_progress_m", mean_prog)

        # Kategori 2: Presisi Lateral
        self.logger.record("lateral/mean_e_lat_m", mean_elat)
        self.logger.record("lateral/max_e_lat_m", max_elat)
        self.logger.record("lateral/max_rear_right_elat_m", max_rr_elat)

        # Kategori 3: Kebebasan Trotoar
        self.logger.record("curb/min_effective_curb_dist_m", min_curb)
        self.logger.record("curb/mean_effective_curb_dist_m", mean_curb)

        # Kategori 4: Performa Kecepatan
        self.logger.record("performance/avg_speed_kmh", avg_spd)
        self.logger.record("performance/speed_tracking_err_kmh", spd_err)

        # Kategori 5: Keselamatan & Insiden Kumulatif
        self.logger.record("safety/cbe_intervention_pct", cbe_ratio_pct)
        self.logger.record("safety/cls_intervention_pct", cls_ratio_pct)
        self.logger.record("safety/collision_events", self.rollout_collision_count)
        self.logger.record("safety/off_track_events", self.rollout_off_track_count)
        self.logger.record("safety/curb_climb_events", self.rollout_curb_climb_count)
        self.logger.record("safety/stuck_events", self.rollout_stuck_count)

        # Cetak Kartu Indikator Kelayakan di Terminal
        status_shield = "LAYAK / OTONOM (<25%)" if shield_ratio_pct < 25.0 else ("TRANSISI (25-50%)" if shield_ratio_pct < 50.0 else "KRITIS / DIDOMINASI PERISAI (>50%)")
        status_curb = "AMAN (>0.2m)" if min_curb >= 0.20 else ("WASPADA (0-0.2m)" if min_curb >= 0.0 else "MELINDAS TROTOAR (<0m)")
        status_lateral = "PRESISI (<0.45m)" if mean_elat < 0.45 else ("CUKUP (<0.7m)" if mean_elat < 0.70 else "LEMAH (>0.7m)")

        print("\n" + "="*72)
        print(f"       HASIL EVALUASI KELAYAKAN PELATIHAN (ROLLOUT #{self.rollout_count})")
        print("="*72)
        print(f" * Rasio Intervensi Perisai : {shield_ratio_pct:5.1f}%  -> [{status_shield}]")
        print(f" * Rekor Jarak Tempuh       : {max_prog:5.1f} m (Rata-rata: {mean_prog:5.1f} m)")
        print(f" * Deviasi Lateral Jalur     : {mean_elat:5.2f} m (Max: {max_elat:.2f} m, RR: {max_rr_elat:.2f} m) -> [{status_lateral}]")
        print(f" * Jarak Bebas Trotoar Min   : {min_curb:5.2f} m  -> [{status_curb}]")
        print(f" * Kepatuhan Kecepatan       : Galat {spd_err:4.1f} km/h (Avg Spd: {avg_spd:4.1f} km/h)")
        print(f" * Total Insiden Rollout    : Tabrakan: {self.rollout_collision_count} | Off-track: {self.rollout_off_track_count} | Curb-climb: {self.rollout_curb_climb_count}")
        print("="*72 + "\n")

        # Reset penampung rollout
        self.rollout_speeds = []
        self.rollout_lateral_errors = []
        self.rollout_rear_right_errors = []
        self.rollout_curb_distances = []
        self.rollout_speed_errors = []
        self.rollout_progress_m = []
        self.rollout_total_steps = 0
        self.rollout_shield_steps = 0
        self.rollout_cbe_steps = 0
        self.rollout_cls_steps = 0
        self.rollout_collision_count = 0
        self.rollout_off_track_count = 0
        self.rollout_curb_climb_count = 0
        self.rollout_stuck_count = 0

def main():
    parser = argparse.ArgumentParser(description="Pelatihan Mandiri (Solo Cruising) PPO pada Jalur Jalan Raya / Pegunungan")
    parser.add_argument("--route", type=str, default="suburban", choices=["suburban", "mountain_castle", "mountain_village"],
                        help="Pilihan rute: suburban, mountain_castle, atau mountain_village")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="Jumlah total langkah pelatihan (default: 5000)")
    parser.add_argument("--ep_steps", type=int, default=2500,
                        help="Maksimum langkah per episode (default: 2500 ~ 4500 meter)")
    parser.add_argument("--speed", type=float, default=80.0,
                        help="Target kecepatan jelajah di jalan lurus (km/jam, default: 80.0)")
    parser.add_argument("--start_offset", type=float, default=48.0,
                        help="Titik awal spawn mobil sepanjang lintasan waypoints (meter, default: 48.0 = jalan utama luar SPBU)")
    parser.add_argument("--model_path", type=str, default="",
                        help="Path model awal (.zip). Kosongkan untuk otomatis melanjutkan dari checkpoint Suburban terbaru")
    parser.add_argument("--scratch", action="store_true",
                        help="Latih model murni dari awal (scratch/0), abaikan seluruh checkpoint sebelumnya")
    parser.add_argument("--reset_timesteps", action="store_true",
                        help="Reset hitungan timesteps dan iterasi ke 0 saat memuat model awal")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="Frekuensi penyimpanan checkpoint otomatis dalam timesteps (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate optimizer PPO (default: 1e-4 = 0.0001, mencegah policy shock)")
    parser.add_argument("--n_steps", type=int, default=1024,
                        help="Kapasitas rollout buffer PPO (default: 1024, menstabilkan GAE dan mencegah buffer poisoning)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Ukuran batch update PPO (default: 64)")
    args, _ = parser.parse_known_args()

    run_name = f"PPO_{args.route.capitalize()}_Solo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("===========================================================")
    print(f"   PELATIHAN SOLO AUTONOMOUS CRUISING: {run_name}          ")
    print("===========================================================")

    # 1. Inisialisasi Lingkungan Tanpa Kendaraan Rintangan (Solo Mode)
    print(f"-> Membuka lingkungan rute '{args.route}' tanpa rintangan (Offset: {args.start_offset:.1f}m, Target: {args.speed:.0f} km/h)...")
    env = BeamNGSuburbanEnv(
        route_type=args.route,
        start_offset_m=args.start_offset,
        has_obstacle=False,
        target_cruise_kmh=args.speed,
        use_yolo=True
    )
    env.max_steps = args.ep_steps

    # 2. Pemilihan Model Awal (Transfer Learning / Continuous Training)
    initial_model = None
    if args.scratch or (args.model_path and args.model_path.lower() in ["none", "scratch", "0"]):
        print("-> Mode Pelatihan dari Awal (--scratch): Mengabaikan seluruh checkpoint lama, melatih model murni dari 0.")
        initial_model = None
    elif args.model_path and os.path.exists(args.model_path):
        initial_model = args.model_path
    else:
        # Otomatis cek apakah sudah ada model Suburban Solo sebelumnya
        existing_suburban = sorted(glob.glob(os.path.join(MODEL_DIR, "PPO_Suburban_Solo_*.zip")))
        if existing_suburban:
            initial_model = existing_suburban[-1]
        elif os.path.exists(BASE_MODEL_PATH):
            initial_model = BASE_MODEL_PATH

    if initial_model:
        print(f"-> Melanjutkan pelatihan / Transfer Learning dari model: {initial_model}")
        print(f"-> Menyetel ulang Rollout Buffer PPO: n_steps={args.n_steps}, batch_size={args.batch_size}, lr={args.lr} (tabel dicetak tiap {args.n_steps} langkah)")
        model = PPO.load(
            initial_model,
            env=env,
            device="cpu",
            tensorboard_log=LOG_DIR,
            verbose=1,
            custom_objects={
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr
            }
        )
    else:
        print(f"-> Membangun model PPO baru pada CPU (LR: {args.lr}, Buffer: {args.n_steps}, Batch: {args.batch_size})...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cpu"
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=MODEL_DIR,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks = CallbackList([SoloMetricsCallback(start_ep=args.start_ep), checkpoint_callback])

    print(f"\n--- Memulai Pelatihan Solo ({args.timesteps} Timesteps, Max {args.ep_steps} steps/ep) ---")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            log_interval=1,
            reset_num_timesteps=(initial_model is None or args.reset_timesteps)
        )
        print("\nPelatihan selesai dengan sukses!")
    except KeyboardInterrupt:
        print("\nPelatihan dihentikan pengguna.")
    finally:
        save_path = os.path.join(MODEL_DIR, f"{run_name}_final")
        model.save(save_path)
        print(f"Model tersimpan ke: {save_path}.zip")
        env.close()

if __name__ == "__main__":
    main()

