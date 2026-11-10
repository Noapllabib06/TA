import numpy as np
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

def normalize_angle(angle_rad):
    """Normalisasi sudut ke rentang [-pi, pi]."""
    while angle_rad > np.pi:
        angle_rad -= 2.0 * np.pi
    while angle_rad < -np.pi:
        angle_rad += 2.0 * np.pi
    return angle_rad

class WaypointManager:
    """
    Mengelola lintasan referensi (waypoints), perhitungan kesalahan lateral & heading,
    kontrol kemudi otomatis menggunakan Speed-Adaptive Stanley Controller (anti-weaving di jalan lurus)
    dengan titik pusat acuan di Pusat Massa / Tengah Kendaraan (Center of Gravity / Mid-Wheelbase),
    serta Profil Kecepatan Kinematis Global (Kinematic Speed Profile) untuk jelajah otonom halus hingga 80+ km/jam.
    """
    def __init__(self, waypoints=None, k_e=1.20, k_soft=0.8, wheelbase=2.7, front_axle_dist=0.0, lateral_offset_m=0.0, target_cruise_kmh=80.0, use_dynamic_reference=False, rear_axle_dist=1.30, front_overhang=0.85, rear_overhang=1.05):
        """
        Args:
            waypoints: Array numpy berbentuk (N, 2) atau (N, 3) koordinat [x, y, (z)]
            k_e: Gain cross-track error pada Stanley Controller (optimal: 1.20)
            k_soft: Konstanta pelunakan kecepatan rendah (m/s, optimal: 0.8)
            wheelbase: Jarak sumbu roda kendaraan (etk800 ~ 2.7m)
            front_axle_dist: Jarak proyeksi titik acuan statis dari CG (jika use_dynamic_reference=False)
            lateral_offset_m: Buffer jarak lateral aman ke arah kiri lajur (m, e.g. 0.30m untuk menjauhi trotoar kanan)
            target_cruise_kmh: Kecepatan jelajah maksimum di jalan lurus (km/jam)
            use_dynamic_reference: Mengaktifkan Dynamic Reference Point (CoM di jalan lurus/flat-out <-> CoRA di tikungan tajam)
            rear_axle_dist: Jarak dari CoM ke sumbu roda belakang (etk800 ~ 1.30m)
            front_overhang: Jarak as roda depan ke bemper depan (etk800 ~ 0.85m)
            rear_overhang: Jarak as roda belakang ke bemper belakang (etk800 ~ 1.05m)
        """
        self.k_e = k_e
        self.k_soft = k_soft
        self.wheelbase = wheelbase
        self.front_axle_dist = front_axle_dist
        self.lateral_offset_m = lateral_offset_m
        self.target_cruise_kmh = target_cruise_kmh
        self.use_dynamic_reference = use_dynamic_reference
        self.rear_axle_dist = rear_axle_dist
        self.front_overhang = front_overhang
        self.rear_overhang = rear_overhang
        self.last_ref_offset = 0.0
        self.last_w_rear = 0.0
        self.smooth_dynamic_offset = 0.0
        self.smooth_corner_offset = 0.0
        self.prev_e_lat = 0.0
        self.waypoints = None
        self.xy = None
        self.segment_headings = None
        self.curvatures = None
        self.speed_profile = None
        self.last_closest_idx = 0
        self.prev_steering = 0.0
        self.steer_filter_alpha = 0.70
        if waypoints is not None:
            self.set_waypoints(waypoints, target_cruise_kmh=target_cruise_kmh)

    def set_waypoints(self, waypoints, target_cruise_kmh=None):
        """Menyimpan waypoints asli (jalur rekaman manusia), menghitung heading segmen, kelengkungan, dan profil kecepatan."""
        if target_cruise_kmh is not None:
            self.target_cruise_kmh = target_cruise_kmh

        self.waypoints = np.array(waypoints, dtype=np.float32)
        if self.waypoints.shape[1] > 2:
            self.xy = np.copy(self.waypoints[:, :2])
        else:
            self.xy = np.copy(self.waypoints)

        # Hitung arah hadap jalan pada setiap segmen
        dx = np.diff(self.xy[:, 0], append=self.xy[-1, 0])
        dy = np.diff(self.xy[:, 1], append=self.xy[-1, 1])
        if len(dx) > 1:
            dx[-1] = dx[-2]
            dy[-1] = dy[-2]
        self.segment_headings = np.arctan2(dy, dx)
        self.last_closest_idx = 0
        self.prev_steering = 0.0
        self.prev_ego_yaw = None
        self.smooth_dynamic_offset = 0.0
        self.smooth_corner_offset = 0.0
        self.prev_e_lat = 0.0
        self.last_bbox_errors = {}

        # Bangun Profil Kecepatan Kinematis Global
        self.build_speed_profile(target_cruise_kmh=self.target_cruise_kmh)

    def build_speed_profile(self, target_cruise_kmh=80.0, a_lat_max=1.8, a_decel_comfort=0.8, a_accel_comfort=1.5, v_min_corner_kmh=6.0):
        """
        Membangun Profil Kecepatan Kinematis Global (Kinematic Speed Profiler)
        sepanjang lintasan waypoints menggunakan metode bidirectional dynamic programming:
        - 1. Batas Kecepatan Statis Berdasarkan Kelengkungan (Kamm's Circle Limit)
             Memaksa deselerasi hingga 5.5 - 6.5 km/jam pada tikungan tajam persimpangan.
        - 2. Backward Pass: Propagasi deselerasi progresif menjelang tikungan (anti-brake check)
        - 3. Forward Pass: Propagasi akselerasi progresif keluar tikungan (anti-power-understeer)
        """
        n_points = len(self.xy)
        if n_points == 0:
            self.speed_profile = np.array([], dtype=np.float32)
            self.curvatures = np.zeros(0, dtype=np.float32)
            return

        self.curvatures = np.zeros(n_points, dtype=np.float32)
        lookahead = 4
        for i in range(n_points):
            i2 = min(i + lookahead, n_points - 1)
            if i == i2:
                continue
            dh = normalize_angle(self.segment_headings[i2] - self.segment_headings[i])
            ds = float(np.linalg.norm(self.xy[i2] - self.xy[i]))
            if ds > 0.1:
                self.curvatures[i] = float(dh / ds)

        v_cruise_max = float(target_cruise_kmh / 3.6)
        v_min_corner = float(v_min_corner_kmh / 3.6)

        # 1. Batas Kecepatan Statis Berdasarkan Kelengkungan (Kamm's Circle Limit)
        v_static = np.full(n_points, v_cruise_max, dtype=np.float32)
        for i in range(n_points):
            k = abs(self.curvatures[i])
            if k > 0.005:
                v_lim = np.sqrt(a_lat_max / k)
                # Batasi kecepatan pada tikungan sesuai radius kurvatur
                if k >= 0.035:
                    v_lim = min(v_lim, float(12.0 / 3.6))  # Tikungan sedang-tajam: wajib <= 12 km/jam
                if k >= 0.070:
                    v_lim = min(v_lim, float(6.5 / 3.6))   # Persimpangan sempit / tikungan 90 derajat: 5.5 - 6.5 km/jam
                v_static[i] = min(v_cruise_max, max(v_min_corner, v_lim))

        # 2. Backward Pass: Propagasi Deselerasi Progresif Menjelang Tikungan
        v_profile = np.copy(v_static)
        for i in range(n_points - 2, -1, -1):
            ds = float(np.linalg.norm(self.xy[i+1] - self.xy[i]))
            v_reachable = np.sqrt(v_profile[i+1]**2 + 2.0 * a_decel_comfort * ds)
            v_profile[i] = min(v_profile[i], v_reachable)

        # 3. Forward Pass: Propagasi Akselerasi Progresif Keluar Tikungan (Anti-Power-Understeer)
        for i in range(1, n_points):
            ds = float(np.linalg.norm(self.xy[i] - self.xy[i-1]))
            v_reachable = np.sqrt(v_profile[i-1]**2 + 2.0 * a_accel_comfort * ds)
            v_profile[i] = min(v_profile[i], v_reachable)

        self.speed_profile = v_profile

    def get_target_speed(self, idx):
        """Mengambil kecepatan target aman (m/s) pada indeks waypoint tertentu."""
        if self.speed_profile is None or len(self.speed_profile) == 0:
            return 10.0
        idx = max(0, min(idx, len(self.speed_profile) - 1))
        return float(self.speed_profile[idx])

    def get_upcoming_curvatures(self, closest_idx, lookahead_distance_m=35.0):
        """
        Mengambil array kelengkungan (curvatures) sepanjang lintasan ke depan
        sejauh lookahead_distance_m dari indeks closest_idx.
        """
        if self.curvatures is None or len(self.curvatures) == 0:
            return np.array([0.0], dtype=np.float32)

        n_points = len(self.curvatures)
        closest_idx = max(0, min(closest_idx, n_points - 1))

        upcoming = []
        cum_dist = 0.0
        for j in range(closest_idx, min(closest_idx + 60, n_points - 1)):
            ds = float(np.linalg.norm(self.xy[j+1] - self.xy[j]))
            cum_dist += ds
            if cum_dist > lookahead_distance_m:
                break
            upcoming.append(abs(float(self.curvatures[j+1])))

        if not upcoming:
            upcoming.append(abs(float(self.curvatures[closest_idx])))

        return np.array(upcoming, dtype=np.float32)

    def detect_sharp_corner_ahead(self, closest_idx, lookahead_m=35.0, kappa_threshold=0.045):
        """
        Mendeteksi apakah terdapat tikungan tajam (kelengkungan >= kappa_threshold)
        di depan kendaraan dalam radius lookahead_m meter sepanjang lintasan.
        """
        if self.curvatures is None or len(self.curvatures) == 0:
            return False

        n_points = len(self.curvatures)
        closest_idx = max(0, min(closest_idx, n_points - 1))
        cum_dist = 0.0

        for j in range(closest_idx, min(closest_idx + 80, n_points - 1)):
            ds = float(np.linalg.norm(self.xy[j+1] - self.xy[j]))
            cum_dist += ds
            if cum_dist > lookahead_m:
                break
            if abs(float(self.curvatures[j+1])) >= kappa_threshold:
                return True

        return False

    def get_dynamic_lookahead_target_speed(self, closest_idx, current_speed_ms, a_decel_comfort=1.2):
        """
        Menghitung batas kecepatan target dinamis berbasis horizon pandang ke depan (Dynamic Braking Lookahead).
        Memindai horizon lintasan proporsional terhadap energi kinetik (v^2 / 2a) ditambah buffer persiapan.
        Jika di depan terdeteksi tikungan tajam, deselerasi dimulai bertahap dan selesai 12-15 meter SEBELUM
        mulut belokan, mencegah keterlambatan pengereman (Braking Horizon Lag) dan overspeed saat melibas apex.
        """
        if self.speed_profile is None or len(self.speed_profile) == 0:
            return 10.0

        n_points = len(self.speed_profile)
        closest_idx = max(0, min(closest_idx, n_points - 1))
        base_v_target = float(self.speed_profile[closest_idx])

        # 1. Jarak pengereman kinetis antisipatif proporsional terhadap kuadrat kecepatan (v^2 / 2a)
        #    Ditambah buffer jarak persiapan masuk tikungan 18.0m
        braking_margin_dist = max(35.0, (current_speed_ms**2) / (2.0 * 1.5) + 18.0)

        cum_dist = 0.0
        min_reachable_speed = base_v_target
        max_k = 0.0
        curve_entry_dist = None

        # Pindai hingga ~80 waypoint ke depan (~150 meter)
        for j in range(closest_idx, min(closest_idx + 80, n_points - 1)):
            ds = float(np.linalg.norm(self.xy[j+1] - self.xy[j]))
            cum_dist += ds
            if cum_dist > braking_margin_dist:
                break

            k = abs(float(self.curvatures[j+1])) if self.curvatures is not None and (j+1) < len(self.curvatures) else 0.0
            if k > max_k:
                max_k = k

            # Titik awal belokan dimulai ketika kurvatur mulai naik (k >= 0.020)
            if curve_entry_dist is None and k >= 0.020:
                curve_entry_dist = cum_dist

            v_ahead = float(self.speed_profile[j+1])
            v_reach = np.sqrt(v_ahead**2 + 2.0 * a_decel_comfort * cum_dist)
            if v_reach < min_reachable_speed:
                min_reachable_speed = v_reach

        # Tentukan batas kecepatan berdasarkan kurvatur terburuk di depan
        if max_k >= 0.055:
            v_cap = float(16.0 / 3.6)
        elif max_k >= 0.035:
            v_cap = float(22.0 / 3.6)
        elif max_k >= 0.020:
            v_cap = float(35.0 / 3.6)
        else:
            v_cap = base_v_target

        v_anticipation = base_v_target
        if curve_entry_dist is not None and v_cap < base_v_target:
            # Beri jeda 12-15 meter SEBELUM kurva dimulai agar mobil sudah mencapai v_cap sebelum bodi masuk belokan
            lead_dist = max(0.0, curve_entry_dist - 14.0)
            v_anticipation = np.sqrt(v_cap**2 + 2.0 * a_decel_comfort * lead_dist)

        return float(min(base_v_target, min_reachable_speed, v_anticipation))

    def find_closest_index(self, pos_xy, search_window=60, update_state=True):
        """
        Mencari indeks waypoint terdekat dengan pencarian terfokus maju.
        Diberikan jendela pencarian adaptif [-5 s.d. +60 indeks] (~120 meter)
        agar tidak pernah tertinggal (lagging) saat kendaraan melaju kencang (80 km/jam).
        """
        n_points = len(self.xy)
        if n_points == 0:
            return 0

        # Cari dari (last_closest_idx - 5) hingga (+search_window)
        start_idx = max(0, self.last_closest_idx - 5)
        end_idx = min(n_points, self.last_closest_idx + search_window)

        sub_xy = self.xy[start_idx:end_idx]
        diffs = sub_xy - pos_xy[:2]
        dists_sq = np.sum(diffs**2, axis=1)
        local_min = int(np.argmin(dists_sq))
        best_idx = start_idx + local_min

        # Jamin indeks selalu bergerak maju secara monoton (tidak mundur melewati rute)
        if best_idx < self.last_closest_idx:
            best_idx = self.last_closest_idx

        if update_state:
            self.last_closest_idx = best_idx
        return best_idx

    def compute_errors(self, ego_pos, ego_yaw, update_state=True, lookahead_chord=1):
        """
        Menghitung cross-track error (e_lat) dan heading error (e_yaw).
        
        Prinsip Geometris Tegak Lurus Bebas Sawtooth:
        - e_lat dihitung murni menggunakan vektor normal segmen lokal (segment_headings[idx]),
          sehingga pergerakan maju (longitudinal) mobil di sepanjang segmen memiliki proyeksi NOL
          terhadap vektor normal. Hal ini sepenuhnya menghilangkan artefak osilasi 15 Hz / gigi gergaji.
        - e_yaw menghitung deviasi arah hadap mobil terhadap lajur jalan lokal atau chord lookahead.
        """
        pos = np.array(ego_pos[:2], dtype=np.float32)
        idx = self.find_closest_index(pos, update_state=update_state)
        target_pt = self.xy[idx]

        # 1. Sudut hadap segmen lokal jalan
        seg_h = float(self.segment_headings[idx])

        # 2. Vektor normal ortogonal murni terhadap arah segmen jalan
        # Vektor tangen: [cos(seg_h), sin(seg_h)]
        # Vektor normal (kanan lajur): [sin(seg_h), -cos(seg_h)]
        normal_vec = np.array([np.sin(seg_h), -np.cos(seg_h)])
        
        # Vektor perpindahan dari waypoint terdekat ke mobil
        vec_to_ego = pos - target_pt

        # Cross-track error bertanda (+ jika di kanan lajur, - jika di kiri lajur)
        e_lat = float(np.dot(vec_to_ego, normal_vec))

        # 3. Heading error: gunakan lookahead chord jika diminta untuk membuang noise diskritisasi
        if lookahead_chord > 1:
            look_idx = min(len(self.xy) - 1, idx + lookahead_chord)
            dx_h = float(self.xy[look_idx, 0] - target_pt[0])
            dy_h = float(self.xy[look_idx, 1] - target_pt[1])
            if (dx_h**2 + dy_h**2) > 0.25:
                path_heading = float(np.arctan2(dy_h, dx_h))
            else:
                path_heading = seg_h
        else:
            path_heading = seg_h

        e_yaw = float(normalize_angle(path_heading - ego_yaw))

        return idx, e_lat, e_yaw

    def compute_steering(self, ego_pos, ego_yaw, speed_ms):
        """
        Menghitung perintah kemudi [-1.0 s.d. 1.0] menggunakan Speed-Adaptive Stanley Controller.
        
        Arsitektur Titik Acuan Pusat Kendaraan (Center of Vehicle / CG / Mid-Wheelbase):
          1. Center of Gravity Tracking: Titik lacak diletakkan di pusat massa kendaraan (front_axle_dist = 0.0)
             di jalan lurus, dan bergeser ke CoRA (-1.30m) HANYA pada tikungan tajam sempit (R <= 28m).
          2. Speed-Adaptive Heading Gain k_yaw(v):
             Meredam sensitivitas heading berlebih pada kecepatan tinggi (v_ref = 8.0 m/s ~ 28.8 km/jam),
             menghilangkan fenomena 'goyang-goyang / limit-cycle fishtailing' di jalan lurus.
          3. Multi-Point Heading Chord Filtering:
             Menghilangkan chatter noise diskritisasi rekaman waypoint pada jalan lurus panjang.
        """
        # 1. Curvature-Adaptive Dynamic Reference Point & Dynamic Lookahead
        pos_raw = np.array(ego_pos[:2], dtype=np.float32)
        idx_cg = self.find_closest_index(pos_raw, update_state=True)
        
        # Dynamic Lookahead:
        k_local = abs(self.get_curvature(idx_cg, lookahead_steps=1))
        if k_local > 0.06:
            lookahead_steps = 1
        elif k_local > 0.02:
            lookahead_steps = 2
        else:
            lookahead_steps = 3

        curv = self.get_curvature(idx_cg, lookahead_steps=lookahead_steps)
        k_abs = abs(curv)
        
        # Ambang batas kelengkungan yang akurat:
        # HANYA aktif pada tikungan tajam nyata (k_flat = 0.012 ~ R=83m, k_sharp = 0.025 ~ R=40m).
        # Pada jalan lurus / kurva landai (k <= 0.012), mobil 100% terkunci stabil pada Center of Mass (CoM).
        k_flat = 0.012
        k_sharp = 0.025
        
        if self.use_dynamic_reference:
            if k_abs <= k_flat:
                w_rear_tgt = 0.0
            elif k_abs >= k_sharp:
                w_rear_tgt = 1.0
            else:
                t = (k_abs - k_flat) / (k_sharp - k_flat)
                w_rear_tgt = float(t * t * (3.0 - 2.0 * t))
            dynamic_offset_tgt = -float(self.rear_axle_dist * w_rear_tgt)
            # Corner Expansion Progresif: 0.28m pada kurva sedang s.d. 0.45m pada tikungan 90° tajam (mengimbangi off-tracking ban belakang)
            expansion_scale = 0.45 if k_abs >= 0.08 else (0.28 + (k_abs / 0.08) * 0.17)
            corner_offset_tgt = -float(np.sign(curv)) * w_rear_tgt * expansion_scale
        else:
            w_rear_tgt = 0.0
            dynamic_offset_tgt = float(self.front_axle_dist)
            corner_offset_tgt = 0.0

        # Rate Limiter Transition (Anti-Jerk saat transisi tikungan ke jalan lurus):
        # Mencegah lonjakan diskontinu e_lat sebesar ~0.5m saat keluar tikungan menuju jalan lurus panjang
        max_rate_offset = 0.20  # Perpindahan halus max 20 cm per step (~0.8 m/s)
        delta_dyn = dynamic_offset_tgt - self.smooth_dynamic_offset
        self.smooth_dynamic_offset += float(np.clip(delta_dyn, -max_rate_offset, max_rate_offset))

        # Asymmetric Corner Offset Decay (Axle Lag Exit Protection):
        # Masuk belokan: cepat (+0.20 per step) untuk segera melindungi ban belakang dari off-tracking
        # Keluar belokan: meluruh perlahan (-0.035 per step ~ 13 step / ~3.8m) agar roda belakang sedan yang
        # tertinggal di belakang CoM sepenuhnya bebas dari bibir beton sebelum sasis kembali ke sumbu tengah!
        delta_cor = corner_offset_tgt - self.smooth_corner_offset
        if abs(corner_offset_tgt) > abs(self.smooth_corner_offset):
            self.smooth_corner_offset += float(np.clip(delta_cor, -max_rate_offset, max_rate_offset))
        else:
            decay_rate = 0.035
            self.smooth_corner_offset += float(np.clip(delta_cor, -decay_rate, decay_rate))

        dynamic_offset = self.smooth_dynamic_offset
        corner_offset = self.smooth_corner_offset
        w_rear = w_rear_tgt

        self.last_ref_offset = dynamic_offset
        self.last_w_rear = w_rear

        # Pada jalan lurus (w_rear == 0), gunakan chord filter 3 langkah untuk membuang noise rekaman waypoint
        lookahead_chord = 1 if w_rear > 0.1 else 3

        if abs(dynamic_offset) > 0.01:
            ref_x = float(ego_pos[0] + dynamic_offset * np.cos(ego_yaw))
            ref_y = float(ego_pos[1] + dynamic_offset * np.sin(ego_yaw))
            ref_pos = np.array([ref_x, ref_y], dtype=np.float32)
            idx, e_lat, e_yaw = self.compute_errors(ref_pos, ego_yaw, update_state=False, lookahead_chord=lookahead_chord)
            _, e_lat_cg, _ = self.compute_errors(pos_raw, ego_yaw, update_state=False, lookahead_chord=lookahead_chord)
        else:
            ref_pos = pos_raw
            idx, e_lat, e_yaw = self.compute_errors(ref_pos, ego_yaw, update_state=False, lookahead_chord=lookahead_chord)
            e_lat_cg = e_lat
        
        # 2a. Speed-Adaptive Heading Gain k_yaw(v)
        # Menjamin redaman kritis optimal (v_ref = 8.0 m/s ~ 28.8 km/jam)
        v_ref = 8.0
        k_yaw = min(1.0, v_ref / max(speed_ms, v_ref))
        heading_term = k_yaw * e_yaw

        # 2b. Yaw-Rate Damping Term (Peredam Kecepatan Rotasi Sudut Geleng)
        # Shockbreaker Kemudi: Menghilangkan osilasi hunting / fish-tailing / ping-pong
        if getattr(self, 'prev_ego_yaw', None) is None:
            self.prev_ego_yaw = ego_yaw
        delta_yaw = normalize_angle(ego_yaw - self.prev_ego_yaw)
        self.prev_ego_yaw = ego_yaw
        dt_est = 0.15 # Estimasi delta waktu antar step (15 physics steps ~ 0.15s)
        yaw_rate = delta_yaw / dt_est
        kd_yaw = 0.18  # Ditingkatkan ke 0.18 untuk meredam osilasi lateral / zigzag
        damping_term = kd_yaw * float(np.clip(yaw_rate, -2.5, 2.5))

        # 3. Estimasi Bounding Box 4 Sudut Kendaraan
        self.last_bbox_errors = self.compute_bounding_box_errors(e_lat_cg, e_yaw)

        # 4. Stanley Cross-Track Error dengan Dynamic Cornering Expansion & Speed-Scheduled Gain
        # Pada kecepatan rendah (< 15 km/jam / 4.17 m/s): k_e tinggi (self.k_e) agar belokan tajam 90° presisi.
        # Pada kecepatan tinggi (> 25 km/jam / 6.94 m/s): k_e meluruh ke 0.50 agar tidak membanting setir di trek lurus.
        target_high_ke = min(self.k_e, 0.50)
        if speed_ms < 4.17:
            effective_ke = self.k_e
        elif speed_ms > 6.94:
            effective_ke = target_high_ke
        else:
            ratio = (speed_ms - 4.17) / (6.94 - 4.17)
            effective_ke = self.k_e - ratio * (self.k_e - target_high_ke)

        e_lat_effective = e_lat + self.lateral_offset_m + corner_offset
        cross_track_term = np.arctan2(effective_ke * e_lat_effective, max(speed_ms, 0.1) + self.k_soft)
        if abs(curv) < 0.005 and abs(e_lat_effective) < 0.035:
            cross_track_term *= (abs(e_lat_effective) / 0.035)

        # 5. Kompensasi umpan-maju kelengkungan (Curvature Feedforward)
        feedforward_term = np.arctan(self.wheelbase * curv)

        # 6. Sudut kemudi roda teoritis (positif = belok kiri dalam koordinat kartesian)
        # Suku redaman rotasi (- damping_term) secara aktif meredam yaw rate berlebih
        steering_angle = heading_term + cross_track_term + feedforward_term - damping_term

        # 7. Normalisasi ke rentang aktuasi BeamNG [-1.0 s.d. 1.0]
        MAX_STEER_RAD = 0.61
        raw_steering = -float(np.clip(steering_angle / MAX_STEER_RAD, -1.0, 1.0))
        
        # 8. Speed-Adaptive Actuator Filter (Anti-Limit-Cycle Weaving di Kecepatan Tinggi):
        # Di kecepatan rendah / tikungan tajam (v <= 10 m/s): alpha = 0.65 (responsif belok tajam).
        # Di kecepatan tinggi (v >= 20 m/s ~ 72-80 km/jam): alpha = 0.35 (redaman optimal, rock-solid, bebas goyang).
        if speed_ms <= 10.0:
            effective_alpha = 0.65
        elif speed_ms >= 20.0:
            effective_alpha = 0.35
        else:
            effective_alpha = 0.65 - 0.30 * ((speed_ms - 10.0) / 10.0)

        cmd_steering = effective_alpha * raw_steering + (1.0 - effective_alpha) * self.prev_steering
        self.prev_steering = cmd_steering

        return cmd_steering, e_lat, e_yaw

    def get_curvature(self, idx, lookahead_steps=5):
        """Menghitung kelengkungan jalan (curvature) pada jarak pandang di depan."""
        if self.waypoints is None or len(self.waypoints) < 3:
            return 0.0
            
        i1 = min(idx, len(self.segment_headings) - 1)
        i2 = min(idx + lookahead_steps, len(self.segment_headings) - 1)
        
        if i1 == i2:
            return 0.0
            
        dh = normalize_angle(self.segment_headings[i2] - self.segment_headings[i1])
        ds = float(np.linalg.norm(self.xy[i2] - self.xy[i1]))
        if ds < 0.1:
            return 0.0
        return float(dh / ds)

    def compute_bounding_box_errors(self, e_lat_cg, e_yaw):
        """
        Menghitung estimasi galat lateral untuk batas dimensi fisik kendaraan (Oriented Bounding Box - OBB):
        1. Tapak Roda (Wheel Ground Contact):
           - Depan Kanan/Kiri (Front-Right / Front-Left)
           - Belakang Kanan/Kiri (Rear-Right / Rear-Left) -> Titik kontak kritis yang melindas trotoar
        2. Ujung Bemper Terluar (Body Envelope Corners):
           - Bemper Depan Kanan/Kiri (Bumper-FR / Bumper-FL)
           - Bemper Belakang Kanan/Kiri (Bumper-RR / Bumper-RL) -> Titik ayunan overhang kritis
        
        Mempertimbangkan parameter asimetris struktur sasis:
        - Wheelbase (LW ~ 2.70m)
        - Front Overhang (LF ~ 0.85m)
        - Rear Overhang (LR ~ 1.05m)
        - Half Width (W_half ~ 0.93m)
        """
        L_half = float(self.wheelbase / 2.0)  # Jarak CoM ke as roda (~1.35 m)
        L_front_bumper = L_half + getattr(self, 'front_overhang', 0.85)  # Jarak CoM ke bemper depan (~2.20 m)
        L_rear_bumper  = L_half + getattr(self, 'rear_overhang', 1.05)   # Jarak CoM ke bemper belakang (~2.40 m)
        W_half = 0.93                                                   # Setengah lebar bodi ETK 800 (~0.93 m)
        
        sin_yaw = float(np.sin(e_yaw))
        cos_yaw = float(np.cos(e_yaw))
        
        return {
            # Tapak Roda (Wheel Ground Contact - Kontak Trotoar)
            "front_right": float(e_lat_cg + L_half * sin_yaw + W_half * cos_yaw),
            "rear_right":  float(e_lat_cg - L_half * sin_yaw + W_half * cos_yaw),
            "front_left":  float(e_lat_cg + L_half * sin_yaw - W_half * cos_yaw),
            "rear_left":   float(e_lat_cg - L_half * sin_yaw - W_half * cos_yaw),
            # Ujung Bodi Terluar (Body Envelope Corners - Rintangan Dinding / Tiang / Bumper Clearance)
            "bumper_front_right": float(e_lat_cg + L_front_bumper * sin_yaw + W_half * cos_yaw),
            "bumper_rear_right":  float(e_lat_cg - L_rear_bumper * sin_yaw + W_half * cos_yaw),
            "bumper_front_left":  float(e_lat_cg + L_front_bumper * sin_yaw - W_half * cos_yaw),
            "bumper_rear_left":   float(e_lat_cg - L_rear_bumper * sin_yaw - W_half * cos_yaw),
        }

    @staticmethod
    def calculate_start_pose(df_or_csv):
        """
        Menghitung posisi awal dan orientasi quaternion secara analitik dari data waypoint,
        memastikan ego vehicle di-spawn 100% menghadap lurus ke arah jalan (e_yaw = 0.0°).
        """
        import pandas as pd
        if isinstance(df_or_csv, str):
            df = pd.read_csv(df_or_csv)
        else:
            df = df_or_csv

        start_pos = (float(df['x'].iloc[0]), float(df['y'].iloc[0]), float(df['z'].iloc[0]) + 0.15)
        
        # Arah segmen pertama
        dx = float(df['x'].iloc[1] - df['x'].iloc[0])
        dy = float(df['y'].iloc[1] - df['y'].iloc[0])
        
        # Di BeamNG: arah maju lokal kendaraan adalah -Y
        # Rotasi theta CCW terhadap +Z menghasilkan:
        # world_fwd_x = -sin(theta), world_fwd_y = -cos(theta)
        # Sehingga: theta = arctan2(-dx, -dy)
        theta = np.arctan2(-dx, -dy)
        
        qz = float(np.sin(theta / 2.0))
        qw = float(np.cos(theta / 2.0))
        start_rot = (0.0, 0.0, qz, qw)
        
        return start_pos, start_rot

    @staticmethod
    def generate_straight_path(start_pos, length=150.0, step=1.0, heading_rad=0.0):
        """Membuat lintasan lurus sintetis."""
        x0, y0 = start_pos[0], start_pos[1]
        n_points = int(length / step)
        dists = np.linspace(0, length, n_points)
        xs = x0 + dists * np.cos(heading_rad)
        ys = y0 + dists * np.sin(heading_rad)
        return np.column_stack([xs, ys])

    @staticmethod
    def generate_curved_path(start_pos, length=150.0, step=1.0, curve_radius=40.0):
        """Membuat lintasan berbelok (S-curve) sintetis untuk pengujian lateral."""
        n_points = int(length / step)
        s = np.linspace(0, length, n_points)
        xs = start_pos[0] - s
        ys = start_pos[1] + curve_radius * np.sin(2 * np.pi * s / length)
        return np.column_stack([xs, ys])
