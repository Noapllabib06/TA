import numpy as np
from beamng_env import BeamNGSlopeShieldEnv
import time

env = BeamNGSlopeShieldEnv()
obs, info = env.reset()

print("\n--- Uji Lingkungan Gymnasium BeamNG (50 Step, Longitudinal 1D) ---")
print(f"Bentuk Observasi Awal: {obs} | Info: {info}\n")

# Uji 50 step aksi longitudinal 1D
for i in range(50):
    action = [0.8]  # Gas 80% (1D)
    obs, reward, terminated, truncated, info = env.step(action)
    
    speed_kmh = obs[0] * 3.6
    dist_lidar = obs[1]
    pitch_deg = np.degrees(obs[2])
    
    status = "SHIELD INTERVENED" if info["shield_triggered"] else "DRL IN CONTROL"
    print(f"Step {i+1:02d} | Speed: {speed_kmh:5.2f} km/h | Jarak: {dist_lidar:5.2f} m | d_safe: {info['d_safe']:5.2f} m | Reward: {reward:6.2f} | Status: {status}")
    
    if terminated or truncated:
        print(f"-> Episode mencapai terminasi di Step {i+1:02d} (Collision: {info.get('collision', False)} | Safe Stop: {info.get('safe_stop', False)}).")
        break
        
    time.sleep(0.02)

env.close()
print("Pengujian selesai.")
