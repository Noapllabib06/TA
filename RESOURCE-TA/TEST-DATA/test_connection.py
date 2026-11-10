from beamngpy import BeamNGpy, Scenario, Vehicle
import numpy as np
import time

BNG_HOME = r"C:\BeamNG\BeamNG.tech.v0.39.4.0"

bng = BeamNGpy('localhost', 64256, home=BNG_HOME)
bng.open(launch=True)

scenario = Scenario('automation_test_track', 'test_actuation')
vehicle = Vehicle('ego_vehicle', model='etk800')

scenario.add_vehicle(
    vehicle, 
    pos=(489.20, 178.71, 132.47), 
    rot_quat=(0, 0, 0.7071, 0.7071)
)

scenario.make(bng)
bng.scenario.load(scenario)
bng.scenario.start()

# Aktifkan transmisi otomatis arcade agar langsung jalan tanpa kopling manual
vehicle.set_shift_mode('arcade')

print("\n--- Fase 1: Akselerasi Maju (Throttle 1.0) ---")
for step in range(25):
    # Kirim kontrol di setiap iterasi agar simulator terus menerima input gas
    vehicle.control(throttle=1.0, steering=0.0, brake=0.0, parkingbrake=0.0)
    
    bng.control.step(20)  # ~0.33 detik per step
    vehicle.sensors.poll()
    
    # 1. Hitung Kecepatan dari vektor vel (vx, vy, vz)
    vel = vehicle.state.get('vel', (0.0, 0.0, 0.0))
    speed_ms = np.linalg.norm(vel)
    speed_kmh = speed_ms * 3.6
    
    # 2. Hitung Sudut Pitch dari vektor arah hadap (dir)
    dir_vec = vehicle.state.get('dir', (1.0, 0.0, 0.0))
    pitch_rad = np.arcsin(np.clip(dir_vec[2], -1.0, 1.0))
    pitch_deg = np.degrees(pitch_rad)
    
    pos = vehicle.state.get('pos', (0.0, 0.0, 0.0))
    print(f"[ACCEL] Step {step+1:02d} | Speed: {speed_kmh:6.2f} km/h | Pitch: {pitch_deg:6.2f}° | Pos Y: {pos[1]:.2f}")

print("\n--- Fase 2: Pengereman Darurat (Brake 1.0) ---")
for step in range(20):
    # Kirim kontrol rem penuh
    vehicle.control(throttle=0.0, steering=0.0, brake=1.0, parkingbrake=0.0)
    
    bng.control.step(20)
    vehicle.sensors.poll()
    
    vel = vehicle.state.get('vel', (0.0, 0.0, 0.0))
    speed_ms = np.linalg.norm(vel)
    speed_kmh = speed_ms * 3.6
    
    dir_vec = vehicle.state.get('dir', (1.0, 0.0, 0.0))
    pitch_rad = np.arcsin(np.clip(dir_vec[2], -1.0, 1.0))
    pitch_deg = np.degrees(pitch_rad)
    
    print(f"[BRAKE] Step {step+1:02d} | Speed: {speed_kmh:6.2f} km/h | Pitch: {pitch_deg:6.2f}°")

bng.close()
print("\nUji akuasi selesai.")