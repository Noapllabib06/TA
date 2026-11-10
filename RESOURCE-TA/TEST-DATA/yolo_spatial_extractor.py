"""
Modul Ekstraktor Spasial Numerik (YOLOv8-seg Inference -> Spatial Metrics)
--------------------------------------------------------------------------
Bertindak sebagai "Mata Otonom" yang terdekopel (Frozen Inference):
1. Menerima citra RGB dari kamera bemper depan kendaraan.
2. Melakukan inferensi segmentasi menggunakan model terlatih 'best.pt'.
3. Menerapkan proyeksi geometri kamera perspektif-ke-darat (Ground Plane Projection).
4. Menghasilkan metrik spasial numerik terstruktur:
   - d_curb  : Jarak lateral terdekat ke bibir trotoar kanan (meter).
   - e_y     : Deviasi lateral terhadap garis tengah jalan Frenet (meter).
   - e_psi   : Deviasi orientasi heading terhadap arah kurvatur jalan (radian).
"""

import os
import time
import numpy as np
import cv2
import torch
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class SpatialMetrics:
    d_curb: float          # Jarak lateral ke trotoar kanan (meter, >0: aman, <=0: melindas)
    e_y: float             # Deviasi lateral dari tengah jalan (meter, +: kanan, -: kiri)
    e_psi: float           # Deviasi sudut hadap terhadap garis jalan (radian)
    road_width: float      # Estimasi lebar jalan efektif (meter)
    is_curb_detected: bool # Flag apakah trotoar terdeteksi dalam jangkauan
    is_road_detected: bool # Flag apakah permukaan jalan terdeteksi
    inference_ms: float    # Waktu komputasi total inferensi + ekstraksi (ms)

class YOLOSpatialExtractor:
    def __init__(
        self,
        model_path: str = None,
        cam_height: float = 0.85,        # Tinggi kamera di bemper depan (meter)
        cam_pitch_deg: float = 12.5,     # Sudut kemiringan menunduk kamera (derajat)
        fov_y_deg: float = 70.0,         # Field of view vertikal kamera (derajat)
        img_res: tuple = (640, 480),     # Resolusi (width, height)
        vehicle_half_width: float = 0.93 # Setengah lebar bodi ETK 800 (meter)
    ):
        if model_path is None:
            # Path default bobot terbaik yang baru dilatih
            candidate_paths = [
                os.path.join(os.path.dirname(__file__), "runs", "segment", "runs_yolo_curb", "yolov8n_suburban_curb", "weights", "best.pt"),
                os.path.join(os.path.dirname(__file__), "runs_yolo_curb", "yolov8n_suburban_curb", "weights", "best.pt"),
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    model_path = cp
                    break
            if model_path is None:
                model_path = os.path.join(os.path.dirname(__file__), "yolov8n-seg.pt")

        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"-> Menginisialisasi YOLOSpatialExtractor (Device: {self.device})...")
        print(f"-> Memuat bobot 'Frozen' dari: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.cam_height = cam_height
        self.cam_pitch_rad = np.radians(cam_pitch_deg)
        self.img_w, self.img_h = img_res
        self.half_w = vehicle_half_width

        # Parameter Intrinsik Kamera Pinhole
        self.f_y = (self.img_h / 2.0) / np.tan(np.radians(fov_y_deg / 2.0))
        self.f_x = self.f_y
        self.c_x = self.img_w / 2.0
        self.c_y = self.img_h / 2.0

        # Memori Halus (Exponential Moving Average Filter) untuk kestabilan sinyal kontrol
        self.smooth_d_curb = 3.5
        self.smooth_e_y = 0.0
        self.smooth_e_psi = 0.0
        self.alpha = 0.65  # Bobot pembaruan EMA (0.65 baru, 0.35 lama)

        # Precompute tabel proyeksi vertikal piksel-ke-jarak longitudinal (Y_fwd)
        # Menjamin kalkulasi proyeksi berjalan instan (< 0.2 ms)
        v_coords = np.arange(self.img_h)
        alpha_y = np.arctan((v_coords - self.c_y) / self.f_y)
        total_depression = self.cam_pitch_rad + alpha_y
        
        # Valid hanya untuk piksel yang mengarah ke aspal di bawah cakrawala (depression > 0.02 rad)
        self.valid_v_mask = total_depression > 0.03
        self.lut_y_fwd = np.full(self.img_h, fill_value=150.0, dtype=np.float32)
        self.lut_y_fwd[self.valid_v_mask] = self.cam_height / np.tan(total_depression[self.valid_v_mask])

    def pixel_to_ground(self, u: np.ndarray, v: np.ndarray):
        """
        Memproyeksikan koordinat piksel (u, v) ke bidang koordinat horizontal kendaraan (X_lat, Y_fwd).
        - Y_fwd: Jarak maju ke depan bumper (meter)
        - X_lat: Jarak lateral terhadap garis tengah bumper (meter, +: Kanan, -: Kiri)
        """
        v_clipped = np.clip(v, 0, self.img_h - 1)
        y_fwd = self.lut_y_fwd[v_clipped]
        x_lat = y_fwd * ((u - self.c_x) / self.f_x)
        return x_lat, y_fwd

    def extract_metrics(self, img_bgr: np.ndarray) -> SpatialMetrics:
        """
        Ekstraksi metrik spasial numerik secara end-to-end dari satu frame citra.
        """
        t0 = time.perf_counter()
        self.last_img_bgr = img_bgr

        # 1. Inferensi Segmentasi YOLOv8 (Frozen Mode)
        results = self.model(img_bgr, verbose=False, device=self.device)[0]
        self.last_results = results

        is_curb_detected = False
        is_road_detected = False
        raw_d_curb = 4.0
        raw_e_y = 0.0
        raw_e_psi = 0.0
        est_road_width = 3.6

        # 2. Periksa apakah terdapat masker segmentasi yang diprediksi
        if results.masks is not None and len(results.masks) > 0:
            classes = results.boxes.cls.cpu().numpy().astype(int)
            masks_data = results.masks.data.cpu().numpy()  # Bentuk: [N, H_mask, W_mask]

            # Satukan masker untuk kelas 0 (curb) dan kelas 1 (road)
            curb_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            road_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

            for idx, cls_id in enumerate(classes):
                # Resize masker ke dimensi asli citra kamera jika diperlukan
                m = masks_data[idx]
                if m.shape != (self.img_h, self.img_w):
                    m = cv2.resize(m, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

                binary_m = (m > 0.5).astype(np.uint8)
                if cls_id == 0:
                    curb_mask = cv2.bitwise_or(curb_mask, binary_m)
                elif cls_id == 1:
                    road_mask = cv2.bitwise_or(road_mask, binary_m)

            # --- A. Ekstraksi Jarak Trotoar Kanan (d_curb) ---
            curb_v, curb_u = np.where(curb_mask > 0)
            if len(curb_v) > 50:
                curb_x, curb_y = self.pixel_to_ground(curb_u, curb_v)
                
                # Filter area pantauan kritis di depan roda kanan:
                # Longitudinal: 0.8m s.d. 6.0m ke depan
                # Lateral: sisi kanan bodi kendaraan (X_lat > 0.2m)
                lookahead_filter = (curb_y >= 0.8) & (curb_y <= 6.0) & (curb_x >= 0.2)
                if np.sum(lookahead_filter) >= 15:
                    filtered_x = curb_x[lookahead_filter]
                    # Jarak bersih bibir trotoar terhadap roda/bodi kanan mobil
                    clearance_to_right_side = np.percentile(filtered_x, 5) - self.half_w
                    raw_d_curb = float(np.clip(clearance_to_right_side, -0.5, 5.0))
                    is_curb_detected = True

            # --- B. Ekstraksi Deviasi Tengah Jalan (e_y) & Sudut Heading (e_psi) ---
            road_v, road_u = np.where(road_mask > 0)
            if len(road_v) > 200:
                is_road_detected = True
                road_x, road_y = self.pixel_to_ground(road_u, road_v)

                # Evaluasi titik tengah jalur pada irisan jarak longitudinal (2m, 3m, 4m, 5m, 6m)
                slice_dist_bins = [2.0, 3.0, 4.0, 5.0, 6.0]
                center_points = []

                for y_target in slice_dist_bins:
                    slice_mask = (road_y >= (y_target - 0.5)) & (road_y <= (y_target + 0.5))
                    if np.sum(slice_mask) >= 20:
                        xs = road_x[slice_mask]
                        x_left = np.percentile(xs, 10)
                        x_right = np.percentile(xs, 90)
                        x_mid = (x_left + x_right) / 2.0
                        center_points.append((x_mid, y_target))
                        if y_target == 3.0:
                            est_road_width = float(x_right - x_left)

                if len(center_points) >= 2:
                    # e_y diukur pada titik ancang-ancang terdekat (~3.0 meter)
                    closest_center_x = center_points[0][0]
                    # Deviasi lateral: jika titik tengah jalan ada di kanan (x > 0), mobil berada di kiri
                    raw_e_y = float(np.clip(-closest_center_x, -2.5, 2.5))

                    # e_psi dihitung dari gradien kemiringan garis tengah jalan (Linear Fit)
                    pts = np.array(center_points)
                    coeffs = np.polyfit(pts[:, 1], pts[:, 0], deg=1)  # X = a*Y + b
                    raw_e_psi = float(np.clip(np.arctan(coeffs[0]), -np.radians(45), np.radians(45)))

        # 3. Penerapan EMA Filter (Low-Pass Filter) untuk Menghilangkan Noise
        self.smooth_d_curb = (self.alpha * raw_d_curb) + ((1.0 - self.alpha) * self.smooth_d_curb)
        self.smooth_e_y    = (self.alpha * raw_e_y)    + ((1.0 - self.alpha) * self.smooth_e_y)
        self.smooth_e_psi  = (self.alpha * raw_e_psi)  + ((1.0 - self.alpha) * self.smooth_e_psi)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metrics = SpatialMetrics(
            d_curb=float(self.smooth_d_curb),
            e_y=float(self.smooth_e_y),
            e_psi=float(self.smooth_e_psi),
            road_width=est_road_width,
            is_curb_detected=is_curb_detected,
            is_road_detected=is_road_detected,
            inference_ms=elapsed_ms
        )
        self.last_metrics = metrics
        return metrics

    def render_annotated_hud(self, img_bgr: np.ndarray = None, metrics: SpatialMetrics = None, results = None) -> np.ndarray:
        """
        Menggambar jendela Heads-Up Display (HUD) visual lengkap:
        - Bounding Box & Polygon Segmentation Mask hasil deteksi YOLOv8-seg (Road & Curb)
        - Panel Telemetri Spasial Numerik (d_curb, e_y, e_psi, FPS/Latensi)
        """
        if img_bgr is None:
            img_bgr = getattr(self, 'last_img_bgr', None)
        if metrics is None:
            metrics = getattr(self, 'last_metrics', None)
        if results is None:
            results = getattr(self, 'last_results', None)

        if img_bgr is None:
            return None

        # 1. Gambar Bounding Box, Masker Segmentasi Warna, dan Label Kelas via Ultralytics
        if results is not None and hasattr(results, 'plot'):
            try:
                annotated = results.plot(conf=True, boxes=True, masks=True, line_width=2)
            except Exception:
                annotated = img_bgr.copy()
        else:
            annotated = img_bgr.copy()

        if metrics is None:
            return annotated

        # 2. Gambar Panel HUD Telemetri Transparan
        panel_w = 425
        panel_h = 140
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (10 + panel_w, 10 + panel_h), (0, 220, 0), 2)

        # Status Kejujuran Deteksi Trotoar (Jujur: Tampilkan apakah AI benar-benar melihat trotoar)
        if not metrics.is_curb_detected:
            curb_color = (180, 180, 180)  # Abu-abu jika tidak ada trotoar / tidak terdeteksi
            curb_text = "Trotoar : [TIDAK TERDETEKSI / LOSS]"
        elif metrics.d_curb <= 0.20:
            curb_color = (0, 0, 255)      # Merah (Batas Shielding)
            curb_text = f"Trotoar : {metrics.d_curb:+.2f} m [KRITIS SHIELD]"
        elif metrics.d_curb <= 0.50:
            curb_color = (0, 200, 255)    # Kuning
            curb_text = f"Trotoar : {metrics.d_curb:+.2f} m [WASPADAI]"
        else:
            curb_color = (0, 255, 0)      # Hijau
            curb_text = f"Trotoar : {metrics.d_curb:+.2f} m [AMAN]"

        cv2.putText(annotated, curb_text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, curb_color, 2)

        # Status Deteksi Drivable Road Surface
        if metrics.is_road_detected:
            road_text = f"Jalan   : TERDETEKSI (Lebar: {metrics.road_width:.1f} m)"
            road_color = (0, 255, 200)
        else:
            road_text = "Jalan   : [TIDAK TERDETEKSI]"
            road_color = (180, 180, 180)
        cv2.putText(annotated, road_text, (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, road_color, 1)

        cv2.putText(annotated, f"e_y: {metrics.e_y:+.2f} m | e_psi: {np.degrees(metrics.e_psi):+.1f} deg", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        fps = 1000.0 / max(metrics.inference_ms, 0.1)
        cv2.putText(annotated, f"YOLOv8-seg | {metrics.inference_ms:.1f} ms ({fps:.0f} FPS) | Frozen GPU", (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        cv2.putText(annotated, "Fail-Safe: Dual Physical Shield Aktif", (20, 134),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

        # 3. Picture-in-Picture (PiP): Tampilkan Citra Kamera Mentah Asli di Pojok Kanan Atas
        try:
            pip_w, pip_h = 180, 135
            raw_pip = cv2.resize(img_bgr, (pip_w, pip_h))
            x_start = annotated.shape[1] - pip_w - 15
            y_start = 10
            annotated[y_start:y_start+pip_h, x_start:x_start+pip_w] = raw_pip
            cv2.rectangle(annotated, (x_start, y_start), (x_start+pip_w, y_start+pip_h), (0, 255, 255), 2)
            cv2.putText(annotated, "RAW CAMERA (ASLI)", (x_start+10, y_start+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
            cv2.putText(annotated, "RAW CAMERA (ASLI)", (x_start+10, y_start+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        except Exception:
            pass

        return annotated

    def get_annotated_hud(self) -> np.ndarray:
        """Mengambil citra HUD teranotasi terbaru untuk ditampilkan pada jendela OpenCV."""
        return self.render_annotated_hud()

    def render_hud(self, img_bgr: np.ndarray, metrics: SpatialMetrics) -> np.ndarray:
        return self.render_annotated_hud(img_bgr, metrics)

def main():
    print("===========================================================")
    print("      UJI BENCHMARK EKSTRAKTOR SPASIAL (YOLOV8-SEG)        ")
    print("===========================================================")

    extractor = YOLOSpatialExtractor()
    val_dir = os.path.join(os.path.dirname(__file__), "dataset_yolo", "images", "val")
    output_preview_dir = os.path.join(os.path.dirname(__file__), "runs_yolo_curb", "spatial_preview")
    os.makedirs(output_preview_dir, exist_ok=True)

    val_images = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".jpg")])
    if not val_images:
        print("Error: Tidak ada gambar di dataset_yolo/images/val!")
        return

    print(f"-> Menguji ekstraksi metrik pada {len(val_images)} citra validasi...")
    latencies = []
    d_curb_list = []
    ey_list = []

    for idx, img_path in enumerate(val_images):
        img_bgr = cv2.imread(img_path)
        metrics = extractor.extract_metrics(img_bgr)
        latencies.append(metrics.inference_ms)
        d_curb_list.append(metrics.d_curb)
        ey_list.append(metrics.e_y)

        # Simpan 10 frame contoh pertama dengan visualisasi HUD
        if idx < 10:
            hud_img = extractor.render_hud(img_bgr, metrics)
            out_file = os.path.join(output_preview_dir, f"preview_{idx:03d}.jpg")
            cv2.imwrite(out_file, hud_img)

        if (idx + 1) % 40 == 0 or (idx + 1) == len(val_images):
            print(f"Frame {idx+1:03d}/{len(val_images)}: d_curb={metrics.d_curb:+.2f}m, e_y={metrics.e_y:+.2f}m, e_psi={np.degrees(metrics.e_psi):+.1f}°, Latency={metrics.inference_ms:.1f}ms")

    print("\n" + "=" * 65)
    print("  HASIL BENCHMARK SPASIAL SELESAI!")
    print(f"  Rata-rata Latensi Inferensi : {np.mean(latencies):.2f} ms per frame (~{1000.0/np.mean(latencies):.0f} FPS)")
    print(f"  Rentang Nilai d_curb        : Min={np.min(d_curb_list):.2f} m, Max={np.max(d_curb_list):.2f} m, Rata2={np.mean(d_curb_list):.2f} m")
    print(f"  Rentang Nilai e_y           : Min={np.min(ey_list):.2f} m, Max={np.max(ey_list):.2f} m")
    print(f"  Sampel HUD Visual Disimpan di: runs_yolo_curb/spatial_preview/")
    print("===========================================================")

if __name__ == "__main__":
    main()

