"""
Skrip Pelatihan YOLOv8-seg untuk Segmentasi Trotoar (Curb) & Jalan di BeamNG.tech
--------------------------------------------------------------------------------
Melatih model YOLOv8 Nano Segmentasi pada GPU NVIDIA RTX 4050 menggunakan dataset
yang telah diekstraksi dari kamera semantik.
"""

import os
import torch
from ultralytics import YOLO

def main():
    print("===========================================================")
    print("      PELATIHAN YOLOV8-SEG: DETEKSI TROTOAR & JALAN        ")
    print("===========================================================")

    # Periksa ketersediaan GPU CUDA
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"-> Menggunakan Perangkat Komputasi: {device_name} (device={device_id})")

    yaml_path = os.path.join(os.path.dirname(__file__), "dataset_yolo", "data.yaml")
    model_weight = os.path.join(os.path.dirname(__file__), "yolov8n-seg.pt")

    if not os.path.exists(yaml_path):
        print(f"Error: File konfigurasi dataset tidak ditemukan di '{yaml_path}'!")
        return

    # Inisialisasi model YOLOv8-seg (Nano)
    print(f"-> Memuat bobot dasar: {model_weight}")
    model = YOLO(model_weight)

    print("\n--- Memulai Proses Pelatihan (Fine-Tuning 40 Epochs) ---")
    print("Target: Mempelajari masker piksel kelas 'curb' dan 'road'.")
    print("-" * 60)

    # Jalankan training
    results = model.train(
        data=yaml_path,
        epochs=40,
        imgsz=640,
        batch=16,
        device=device_id,
        workers=2,              # 2 workers aman untuk multiprocessing Windows
        project="runs_yolo_curb",
        name="yolov8n_suburban_curb",
        exist_ok=True,
        save=True,
        plots=True
    )

    print("\n===========================================================")
    print("  PELATIHAN SELESAI DENGAN SUKSES!")
    print(f"  Bobot Model Terbaik (Best Weights) Tersimpan di:")
    print(f"  {results.save_dir}\\weights\\best.pt")
    print("===========================================================")

if __name__ == "__main__":
    main()

