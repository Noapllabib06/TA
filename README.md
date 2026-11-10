# Autonomous Driving Menggunakan Hierarchical Safe Reinforcement Learning di BeamNG.tech

Repository ini berisi kode sumber, modul simulasi, pengawas keselamatan (*Safety Shield*), dan algoritma kendali otonom untuk penelitian **Tugas Akhir (S1 Teknik Informatika)**:
> **"Implementasi Hierarchical Safe Reinforcement Learning untuk Sistem Kemudi Otonom pada Lingkungan Perkotaan dan Pegunungan di BeamNG.tech"**

---

## 🏛️ Arsitektur Sistem

Sistem dirancang menggunakan pendekatan hierarkis (*Hierarchical Autonomous Control Architecture*):

1. **Kendali Longitudinal (PPO 1D - Deep RL):**
   * Mengatur aksi gas (*throttle*) dan rem (*brake*) $a_t \in [-1.0, 1.0]$.
   * State observasi ringkas: $[v_{\text{ego}}, d_{\text{lead}}, \theta_{\text{pitch}}]$.
   * Profil kecepatan kinematis global berbasis batas kurvatur jalan (*Kamm's Friction Circle Limit*).

2. **Kendali Lateral (Speed-Adaptive Stanley Controller):**
   * Pelacakan garis tengah jalur (*lane centering*) dengan kompensasi sudut hadap (*yaw error* $e_{\psi}$) dan deviasi lateral ($e_{\text{lat}}$).
   * **Dynamic Reference Point:** Transisi dinamis titik acuan dari *Center of Mass* (CoM) pada jalan lurus ke *Center of Rear Axle* (CoRA) pada tikungan tajam untuk mengeliminasi pemotongan batu trotoar (*anti-rear-wheel cut-in*).

3. **Pengawas Keselamatan Terpadu (*Integrated Safety Shield*):**
   * **Control Barrier Enforcer (CBE) - Prioritas Utama:** Menjamin batas keselamatan fisik mutlak ($h(x) \ge 0$) pada koridor jalan.
   * **Control Lyapunov Stabilizer (CLS) - Prioritas Sekunder:** Meredam laju setir (*adaptive slew-rate limiter*) dan mengunci kecepatan saat berbelok tajam (*cornering governor*).
   * **Escape Crawl Mechanism:** Mencegah kebuntuan (*deadlock*) pada manuver lambat di dekat batu trotoar.

4. **Persepsi & Evaluasi Multi-Modal:**
   * **YOLOv8-seg:** Segmentasi area jalan (*drivable area*) dan deteksi trotoar.
   * **Ground-Truth Multi-Modal Observer:** Sensor LiDAR proaktif, raycast 16 arah, dan sensor kerusakan deformasi sasis.

---

## 📁 Struktur Direktori Utama

```text
├── RESOURCE-TA/
│   └── TEST-DATA/
│       ├── beamng_suburban_env.py      # Lingkungan simulasi Gymnasium BeamNG
│       ├── waypoint_manager.py         # Pengelola lintasan, Stanley, & profil kecepatan
│       ├── train_suburban_solo.py      # Skrip pelatihan PPO + Callback Evaluasi Kelayakan
│       ├── test_waypoints.py           # Unit testing matematis kendali & geometri
│       ├── waypoints_suburban_town.csv # Dataset lintasan waypoint rute Suburban
│       ├── models_saved/               # Checkpoint bobot PPO (.zip)
│       └── tensorboard_logs/           # Log telemetri TensorBoard
├── yolov8n-seg.pt                      # Bobot model segmentasi YOLOv8-seg
├── yolov8n.pt                          # Bobot model deteksi objek YOLOv8
└── README.md
```

---

## 🚀 Cara Menjalankan Pelatihan

### 1. Prasyarat
* Python 3.10 / 3.11
* BeamNG.tech (Simulator Fisika Kendaraan)
* PyTorch, Stable-Baselines3, Gymnasium, BeamNGpy, Ultralytics YOLOv8, OpenCV, NumPy, Pandas

### 2. Menjalankan Unit Testing Kendali
```powershell
python RESOURCE-TA/TEST-DATA/test_waypoints.py
```

### 3. Memulai Pelatihan Mandiri (Solo Cruising)
```powershell
cd RESOURCE-TA/TEST-DATA
python train_suburban_solo.py `
    --route suburban `
    --speed 80 `
    --start_offset 48.0 `
    --timesteps 10000 `
    --ep_steps 2500 `
    --n_steps 1024 `
    --batch_size 64 `
    --lr 0.0001
```

---

## 📊 Metrik Evaluasi Kelayakan Pelatihan

Skrip pelatihan secara otomatis mencetak kartu evaluasi status setiap 1.024 langkah:
* **`feasibility/shield_ratio_pct`**: Rasio intervensi perisai keselamatan (Target: $< 25\%$).
* **`curb/min_effective_curb_dist_m`**: Jarak bebas terdekat roda ke batu trotoar (Target: $> 0.20\text{ m}$).
* **`lateral/mean_e_lat_m`**: Deviasi lateral rata-rata (Target: $< 0.45\text{ m}$).
* **`train/explained_variance`**: Stabilitas fungsi nilai *Critic* (Target: $> 0.30$).

