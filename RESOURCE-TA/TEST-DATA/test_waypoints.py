import numpy as np
from waypoint_manager import WaypointManager, normalize_angle

def test_straight_path():
    print("--- Test 1: Straight Path Tracking ---")
    start = (100.0, 50.0)
    # Jalur lurus ke arah sumbu +X (heading = 0.0)
    path = WaypointManager.generate_straight_path(start, length=100.0, step=1.0, heading_rad=0.0)
    wm = WaypointManager(path)
    
    # Kasus A: Mobil tepat di tengah jalur dan menghadap lurus
    ego_pos = (120.0, 50.0)
    ego_yaw = 0.0
    steer, e_lat, e_yaw = wm.compute_steering(ego_pos, ego_yaw, speed_ms=10.0)
    print(f"Kasus A (Tepat di jalur)   -> Steer: {steer:+.4f} | e_lat: {e_lat:+.4f} m | e_yaw: {e_yaw:+.4f} rad")
    assert abs(steer) < 0.01, "Mobil di tengah lajur harus menghasilkan steering ~0.0"

    # Kasus B: Mobil melenceng 1.5 meter ke kiri (y = 51.5)
    ego_pos = (120.0, 51.5)
    ego_yaw = 0.0
    steer, e_lat, e_yaw = wm.compute_steering(ego_pos, ego_yaw, speed_ms=10.0)
    print(f"Kasus B (Melenceng ke kiri) -> Steer: {steer:+.4f} | e_lat: {e_lat:+.4f} m | e_yaw: {e_yaw:+.4f} rad")
    # Mobil harus mengarahkan kemudi ke kanan (+steer di BeamNG menuju y=50)
    assert steer > 0.0, "Kemudi harus aktif memutar ke kanan (+steer di BeamNG) untuk mengoreksi kesalahan lateral kiri"

    # Kasus C: Mobil miring ke kanan (heading = -0.2 rad)
    ego_pos = (120.0, 50.0)
    ego_yaw = -0.2
    steer, e_lat, e_yaw = wm.compute_steering(ego_pos, ego_yaw, speed_ms=10.0)
    print(f"Kasus C (Miring -0.2 rad)   -> Steer: {steer:+.4f} | e_lat: {e_lat:+.4f} m | e_yaw: {e_yaw:+.4f} rad")
    assert steer < 0.0, "Kemudi harus memutar ke kiri (-steer di BeamNG) untuk mensejajarkan heading"
    print("-> Test 1 Straight Path: PASSED!\n")

def test_curved_path():
    print("--- Test 2: S-Curved Path Tracking ---")
    start = (500.0, 200.0)
    curved_path = WaypointManager.generate_curved_path(start, length=200.0, step=1.0, curve_radius=30.0)
    wm = WaypointManager(curved_path)
    
    # Uji 5 titik di sepanjang lintasan
    for dist in [10, 40, 80, 120, 160]:
        test_pos = curved_path[dist] + np.array([0.0, 0.5]) # Beri offset 0.5m
        target_heading = wm.segment_headings[dist]
        steer, e_lat, e_yaw = wm.compute_steering(test_pos, target_heading, speed_ms=12.0)
        curv = wm.get_curvature(dist)
        print(f"Jarak {dist:3d}m | Steer: {steer:+.3f} | e_lat: {e_lat:+.3f} m | Curvature: {curv:+.4f} rad/m")

    print("-> Test 2 S-Curved Path: PASSED!\n")

def test_ackermann_offtracking_and_bounding_box():
    print("--- Test 3: Ackermann Off-Tracking & Bounding Box ---")
    start = (100.0, 50.0)
    path = WaypointManager.generate_straight_path(start, length=50.0, step=1.0)
    wm = WaypointManager(path)
    
    # Uji Multi-Point Bounding Box saat mobil di tengah lajur
    bbox = wm.compute_bounding_box_errors(e_lat_cg=0.0, e_yaw=0.0)
    print(f"BBox Netral -> FR: {bbox['front_right']:+.2f}m | RR: {bbox['rear_right']:+.2f}m | FL: {bbox['front_left']:+.2f}m | RL: {bbox['rear_left']:+.2f}m")
    assert abs(bbox['rear_right'] - 0.93) < 0.01, "Sisi kanan harus berjarak +0.93m saat e_lat=0"
    
    # Uji saat mobil yaw miring ke kanan (-0.1 rad saat menikung ke kanan)
    bbox_turn = wm.compute_bounding_box_errors(e_lat_cg=0.0, e_yaw=-0.1)
    print(f"BBox Belok Kanan (-0.1 rad) -> RR: {bbox_turn['rear_right']:+.2f}m (Potong dalam: {bbox_turn['rear_right'] - bbox['rear_right']:+.3f}m)")
    assert bbox_turn['rear_right'] > bbox['rear_right'], "Roda belakang kanan harus memotong lebih dalam ke kanan"
    print("-> Test 3 Ackermann Off-Tracking: PASSED!\n")

def test_dynamic_reference_point():
    print("--- Test 4: Dynamic Curvature-Adaptive Reference Point (CoM <-> CoRA) ---")
    start = (100.0, 50.0)
    
    # 1. Lintasan Lurus (k = 0.0)
    straight_path = WaypointManager.generate_straight_path(start, length=50.0, step=1.0)
    wm_straight = WaypointManager(straight_path, use_dynamic_reference=True)
    wm_straight.compute_steering((120.0, 50.0), ego_yaw=0.0, speed_ms=20.0)
    print(f"Jalan Lurus    -> Offset: {wm_straight.last_ref_offset:+.3f}m | Bobot Belakang: {wm_straight.last_w_rear*100:3.0f}% (Harus 100% CoM = 0.000m)")
    assert abs(wm_straight.last_ref_offset) < 0.001, "Jalan lurus harus menghasilkan offset 0.0m (murni CoM)"

    # 2. Lintasan Tajam S-Curve (Radius 25m -> Curvature ~0.04)
    sharp_path = WaypointManager.generate_curved_path(start, length=100.0, step=1.0, curve_radius=40.0)
    wm_sharp = WaypointManager(sharp_path, use_dynamic_reference=True)
    # Evaluasi pada titik tikungan puncak (dist ~ 25) setelah transisi halus beberapa langkah
    for _ in range(8):
        wm_sharp.compute_steering(sharp_path[25], ego_yaw=wm_sharp.segment_headings[25], speed_ms=8.0)
    print(f"Tikungan Tajam -> Offset: {wm_sharp.last_ref_offset:+.3f}m | Bobot Belakang: {wm_sharp.last_w_rear*100:3.0f}% (Harus 100% CoRA = -1.300m)")
    assert wm_sharp.last_ref_offset <= -1.20, "Tikungan tajam harus menggeser acuan ke sumbu belakang (CoRA ~ -1.30m)"
    print("-> Test 4 Dynamic Reference Point: PASSED!\n")

def test_dynamic_braking_lookahead():
    print("--- Test 5: Dynamic Braking Lookahead (Anti-Horizon Lag) ---")
    start = (100.0, 50.0)
    # Lintasan gabungan: 400 meter lurus, lalu tikungan radius 20 meter (kelengkungan tajam)
    straight = WaypointManager.generate_straight_path(start, length=400.0, step=1.0, heading_rad=0.0)
    last_pt = straight[-1]
    curve = WaypointManager.generate_curved_path(last_pt, length=60.0, step=1.0, curve_radius=20.0)
    full_path = np.vstack([straight, curve[1:]])
    wm = WaypointManager(full_path, target_cruise_kmh=80.0)

    # 1. Pada awal lintasan lurus (dist 50m): kecepatan target harus maksimal jelajah (80 km/jam)
    v_early = wm.get_dynamic_lookahead_target_speed(closest_idx=50, current_speed_ms=22.2) * 3.6
    print(f"Jarak 50m (Jelajah Lurus Awal)    -> Target Kecepatan: {v_early:.1f} km/jam (Cruise 80 km/jam)")
    assert v_early >= 75.0, "Jauh dari tikungan harus tetap jelajah cepat"

    # 2. Pada jarak 300m (100 meter sebelum tikungan pada dist 400m):
    # Dynamic lookahead harus sudah mulai menurunkan target kecepatan secara progresif
    v_anticipate = wm.get_dynamic_lookahead_target_speed(closest_idx=300, current_speed_ms=18.0) * 3.6
    print(f"Jarak 300m (100m sebelum tikungan)-> Target Kecepatan: {v_anticipate:.1f} km/jam (Deselerasi Halus)")
    assert v_anticipate < 50.0, "Pada jarak 100m menjelang tikungan tajam, target kecepatan harus sudah turun"

    # 3. Pada mulut tikungan (dist 395m): target kecepatan harus sudah rendah dan aman
    v_corner = wm.get_dynamic_lookahead_target_speed(closest_idx=395, current_speed_ms=6.0) * 3.6
    print(f"Jarak 395m (Mulut Tikungan)       -> Target Kecepatan: {v_corner:.1f} km/jam (Batas Cornering)")
    assert v_corner <= 20.0, "Di mulut tikungan tajam, kecepatan target harus aman (<= 20 km/jam)"
    print("-> Test 5 Dynamic Braking Lookahead: PASSED!\n")

def test_upcoming_curvatures():
    print("--- Test 6: Upcoming Curvatures Lookahead Horizon ---")
    start = (100.0, 50.0)
    straight = WaypointManager.generate_straight_path(start, length=200.0, step=1.0, heading_rad=0.0)
    last_pt = straight[-1]
    curve = WaypointManager.generate_curved_path(last_pt, length=60.0, step=1.0, curve_radius=15.0)
    full_path = np.vstack([straight, curve[1:]])
    wm = WaypointManager(full_path, target_cruise_kmh=80.0)

    # 1. Pada dist 50m: jalan lurus di depan sejauh 35m harus memiliki kelengkungan ~0.0
    curvs_straight = wm.get_upcoming_curvatures(closest_idx=50, lookahead_distance_m=35.0)
    print(f"Jarak 50m (Lurus)       -> Max upcoming curvature: {np.max(curvs_straight):.4f} rad/m")
    assert np.max(curvs_straight) < 0.005, "Jalan lurus tidak boleh mendeteksi kelengkungan tajam"

    # 2. Pada dist 180m (20m sebelum tikungan tajam pada 200m): harus mendeteksi kelengkungan tajam di depan
    curvs_ahead = wm.get_upcoming_curvatures(closest_idx=180, lookahead_distance_m=35.0)
    max_k = np.max(curvs_ahead)
    print(f"Jarak 180m (20m jelang) -> Max upcoming curvature: {max_k:.4f} rad/m (Harus terdeteksi >0.05)")
    assert max_k >= 0.05, "Harus mendeteksi tikungan tajam dalam horizon 35m"
    print("-> Test 6 Upcoming Curvatures: PASSED!\n")

def test_detect_sharp_corner_ahead():
    print("--- Test 7: Detect Sharp Corner Ahead (Kinetic Braking Margin) ---")
    start = (100.0, 50.0)
    straight = WaypointManager.generate_straight_path(start, length=200.0, step=1.0, heading_rad=0.0)
    last_pt = straight[-1]
    curve = WaypointManager.generate_curved_path(last_pt, length=60.0, step=1.0, curve_radius=15.0)
    full_path = np.vstack([straight, curve[1:]])
    wm = WaypointManager(full_path, target_cruise_kmh=80.0)

    # 1. Pada dist 50m: tidak ada tikungan tajam dalam radius 40m
    is_sharp_early = wm.detect_sharp_corner_ahead(closest_idx=50, lookahead_m=40.0, kappa_threshold=0.045)
    print(f"Jarak 50m (Lurus)       -> Tikungan tajam terdeteksi: {is_sharp_early} (Harus False)")
    assert not is_sharp_early, "Di jalan lurus tidak boleh mendeteksi tikungan tajam"

    # 2. Pada dist 175m (25m sebelum tikungan): harus mendeteksi tikungan tajam
    is_sharp_ahead = wm.detect_sharp_corner_ahead(closest_idx=175, lookahead_m=35.0, kappa_threshold=0.045)
    print(f"Jarak 175m (25m jelang) -> Tikungan tajam terdeteksi: {is_sharp_ahead} (Harus True)")
    assert is_sharp_ahead, "Pada jarak 25m sebelum tikungan tajam harus mendeteksi True"
    print("-> Test 7 Detect Sharp Corner Ahead: PASSED!\n")

if __name__ == "__main__":
    test_straight_path()
    test_curved_path()
    test_ackermann_offtracking_and_bounding_box()
    test_dynamic_reference_point()
    test_dynamic_braking_lookahead()
    test_upcoming_curvatures()
    test_detect_sharp_corner_ahead()
    print("Semua uji matematis WaypointManager & Stanley Controller BERHASIL!")

