"""
Skrip Konversi Anotasi Semantik BeamNG.tech ke Format Poligon YOLOv8-seg (.txt)
------------------------------------------------------------------------------
Membaca citra masker semantik PNG dari folder 'dataset_raw/annotations',
mengekstraksi kontur poligon material trotoar (SIDEWALK) dan jalan (STREET),
serta menyusun struktur dataset siap-latih YOLOv8-seg lengkap dengan 'data.yaml'.
"""

import os
import glob
import cv2
import numpy as np
import shutil

RAW_IMG_DIR = os.path.join(os.path.dirname(__file__), "dataset_raw", "images")
RAW_ANN_DIR = os.path.join(os.path.dirname(__file__), "dataset_raw", "annotations")
YOLO_BASE = os.path.join(os.path.dirname(__file__), "dataset_yolo")

# Konfigurasi Spektrum Warna Ground-Truth dari 'tech/annotations.json'
# Format OpenCV: BGR [Blue, Green, Red]
CURB_RANGES = [
    # SIDEWALK: RGB [89, 118, 155] -> BGR [155, 118, 89]
    (np.array([145, 108, 79], dtype=np.uint8), np.array([165, 128, 99], dtype=np.uint8)),
    # SIDEBARS / Pembatas beton: RGB [31, 74, 214] -> BGR [214, 74, 31]
    (np.array([204, 64, 21], dtype=np.uint8), np.array([224, 84, 41], dtype=np.uint8))
]

ROAD_RANGES = [
    # ASPHALT (Material Utama Jalan East Coast USA): RGB [128, 128, 128] -> BGR [128, 128, 128]
    (np.array([120, 120, 120], dtype=np.uint8), np.array([135, 135, 135], dtype=np.uint8)),
    # STREET (Persimpangan / Aspal Gelap): RGB [255, 0, 0] -> BGR [0, 0, 255]
    (np.array([0, 0, 245], dtype=np.uint8), np.array([10, 10, 255], dtype=np.uint8)),
    # COBBLESTONE (Paving/Batu Kota): RGB [32, 64, 196] -> BGR [196, 64, 32]
    (np.array([186, 54, 22], dtype=np.uint8), np.array([206, 74, 42], dtype=np.uint8)),
    # SOLID_LINE (Garis Marka Utuh): RGB [255, 196, 128] -> BGR [128, 196, 255]
    (np.array([118, 186, 245], dtype=np.uint8), np.array([138, 206, 255], dtype=np.uint8)),
    # DASHED_LINE (Garis Marka Putus): RGB [196, 196, 256] -> BGR [255, 196, 196]
    (np.array([245, 186, 186], dtype=np.uint8), np.array([255, 206, 206], dtype=np.uint8)),
    # ZEBRA_CROSSING: RGB [255, 128, 128] -> BGR [128, 128, 255]
    (np.array([118, 118, 245], dtype=np.uint8), np.array([138, 138, 255], dtype=np.uint8))
]

def main():
    print("===========================================================")
    print("      KONVERTER MASKER SEMANTIK -> FORMAT POLIGON YOLO     ")
    print("===========================================================")

    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_BASE, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_BASE, 'labels', split), exist_ok=True)

    ann_files = sorted(glob.glob(os.path.join(RAW_ANN_DIR, "*.png")))
    total_files = len(ann_files)

    if total_files == 0:
        print(f"Error: Tidak ditemukan file anotasi semantik di '{RAW_ANN_DIR}'!")
        print("Pastikan Anda sudah menjalankan 'collect_semantic_dataset.py' terlebih dahulu.")
        return

    print(f"-> Terdeteksi {total_files} file anotasi semantik.")
    print("-> Konfigurasi Kelas Target: [0] curb (SIDEWALK/SIDEBARS), [1] road (ASPHALT/STREET/MARKING)")

    val_split_ratio = 0.2  # 80% Train, 20% Validation
    processed_count = 0
    polygon_count = 0

    for idx, ann_path in enumerate(ann_files):
        base_name = os.path.basename(ann_path).replace('.png', '')
        img_path = os.path.join(RAW_IMG_DIR, f"{base_name}.jpg")

        if not os.path.exists(img_path):
            continue

        split = 'val' if (idx % int(1.0 / val_split_ratio) == 0) else 'train'

        ann_bgr = cv2.imread(ann_path)
        if ann_bgr is None:
            continue

        h, w, _ = ann_bgr.shape
        label_lines = []

        # Ekstraksi poligon untuk setiap kelas target
        tasks = [
            (0, CURB_RANGES, 80),   # Class 0: curb (min_area: 80 px)
            (1, ROAD_RANGES, 250)   # Class 1: road (min_area: 250 px)
        ]

        for class_id, ranges, min_area in tasks:
            # Gabungkan seluruh spektrum warna untuk kelas ini
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                sub_mask = cv2.inRange(ann_bgr, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, sub_mask)

            # Temukan kontur batas
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue

                # Aproksimasi kurva poligon (Ramer-Douglas-Peucker) untuk kompresi verteks
                epsilon = 0.0035 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) < 3:
                    continue

                # Normalisasi koordinat poligon ke skala [0.0, 1.0]
                coords = []
                for pt in approx:
                    norm_x = float(pt[0][0]) / w
                    norm_y = float(pt[0][1]) / h
                    coords.extend([f"{norm_x:.5f}", f"{norm_y:.5f}"])

                label_lines.append(f"{class_id} " + " ".join(coords) + "\n")
                polygon_count += 1

        # Tulis berkas label .txt
        label_txt_path = os.path.join(YOLO_BASE, 'labels', split, f"{base_name}.txt")
        with open(label_txt_path, 'w') as f:
            f.writelines(label_lines)

        # Salin citra asli ke folder dataset YOLO
        target_img_path = os.path.join(YOLO_BASE, 'images', split, f"{base_name}.jpg")
        shutil.copy2(img_path, target_img_path)

        processed_count += 1

    # Buat file konfigurasi data.yaml otomatis
    yaml_content = f"""# Konfigurasi Dataset YOLOv8-seg untuk Kendali Otonom BeamNG.tech
path: {os.path.abspath(YOLO_BASE).replace(os.sep, '/')}
train: images/train
val: images/val

names:
  0: curb
  1: road
"""
    yaml_path = os.path.join(YOLO_BASE, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("\n" + "=" * 65)
    print(f"  BERHASIL MENGONVERSI DATASET KE FORMAT YOLOV8-SEG!")
    print(f"  Total Pasangan Gambar : {processed_count} frame")
    print(f"  Total Poligon Tercipta : {polygon_count} poligon")
    print(f"  File Konfigurasi YAML  : {yaml_path}")
    print("=" * 65)
    print("\nPerintah untuk memulai pelatihan YOLOv8 Nano Segmentasi:")
    print(f"yolo segment train data=\"{yaml_path}\" model=yolov8n-seg.pt epochs=50 imgsz=640 batch=16 device=0\n")

if __name__ == "__main__":
    main()

