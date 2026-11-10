"""
Skrip Batch Downloader & Auto-Merger untuk Semua Dataset Roboflow Pilihan
------------------------------------------------------------------------
Secara otomatis:
1. Mengunduh dataset-dataset dari Roboflow Universe satu per satu.
2. Menyimpannya ke folder modular yang sesuai di 'Dataset/<Category>/<Project>'.
3. Melewati (skip) proyek yang 404/gagal tanpa menghentikan proses.
4. Menjalankan 'compile_dataset.py' untuk remapping kelas (0: curb, 1: road).
5. Memberikan laporan detail mana yang berhasil dan mana yang tidak bisa.
"""

import os
import sys
import shutil
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH)

from compile_dataset import compile_all

# 11 Dataset Pilihan Pengguna (Nomor 2 yang berupa bounding box kotak sudah dikeluarkan)
DATASET_URLS = [
    ("lane1234", "lane-k8801", "Line", "rf_lane_"),
    ("road-segmentation", "road_segmentationv1.1", "Road", "rf_roadseg1_"),
    ("curbs", "curb-jk8ge", "Curb", "rf_curb1_"),
    ("vspace", "curb-weau3", "Curb", "rf_curb2_"),
    ("rad-khosh-kqqsc", "curb-sizeall", "Curb", "rf_curb3_"),
    ("autonomousmobilerobotics", "road-segmentation-eww07", "Road", "rf_amr_"),
    ("cvml", "road-segmentation-rqcuw", "Road", "rf_cvml_"),
    ("car-racing-f4xxz", "road-segmentation-q6nmf", "Road", "rf_carrace_"),
    ("workspace-aabqj", "road-segmentation-ipcbf", "Road", "rf_ipcbf_"),
    ("internship-tvkfo", "road-lines-segmentation", "Line", "rf_lines_"),
    ("goigaxeurc", "urc-road-segmentation", "Road", "rf_urc_"),
]

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "Dataset")

def main():
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY tidak ditemukan di .env!")
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] Paket 'roboflow' belum terpasang.")
        return

    rf = Roboflow(api_key=api_key)

    print("=========================================================================================")
    print("      BATCH DATASET DOWNLOADER & MODULAR BUILDER UNTUK YOLOV8-SEG                        ")
    print("=========================================================================================")
    print(f"Total Dataset yang Akan Diuji : {len(DATASET_URLS)} proyek")
    print(f"Folder Target Kategori        : {DATASET_ROOT}")
    print("-----------------------------------------------------------------------------------------")

    results_table = []

    for idx, (ws_name, proj_name, category, prefix) in enumerate(DATASET_URLS, 1):
        target_proj_dir = os.path.join(DATASET_ROOT, category, proj_name)
        print(f"\n[{idx:02d}/{len(DATASET_URLS)}] Memproses: {ws_name}/{proj_name} (Kategori: {category}) ...")

        try:
            ws = rf.workspace(ws_name)
            proj = ws.project(proj_name)
            
            # Cek versi terbaru (proj.versions() adalah fungsi method)
            try:
                all_vers = proj.versions()
                if all_vers:
                    ver = all_vers[-1]
                else:
                    ver = proj.version(1)
            except Exception:
                ver = proj.version(1)

            ver_num = getattr(ver, 'version', 1)
            print(f"  -> Mengunduh versi {ver_num} dalam format yolov8 ke Dataset/{category}/{proj_name} ...")
            ver.download("yolov8", location=target_proj_dir, overwrite=True)

            print(f"  -> [BERHASIL] Dataset {proj_name} sukses diunduh!")
            results_table.append((idx, proj_name, category, "BERHASIL", f"Tersimpan di Dataset/{category}/{proj_name}"))

        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "NotFound" in err_msg or "not found" in err_msg.lower():
                status = "GAGAL (404 Not Found)"
                alasan = "Halaman dataset tidak ditemukan di Roboflow"
            elif "403" in err_msg or "401" in err_msg or "unauthorized" in err_msg.lower():
                status = "GAGAL (Private / Unauthorized)"
                alasan = "Proyek privat, akun tidak memiliki izin akses"
            else:
                status = "GAGAL"
                alasan = err_msg[:45]

            print(f"  -> [{status}] {alasan}")
            results_table.append((idx, proj_name, category, status, alasan))

    # Kompilasi seluruh dataset yang berhasil masuk ke dataset_yolo
    print("\n\n=========================================================================================")
    print("          MENJALANKAN KOMPILASI & SMART CLASS REMAPPING KE DATASET_YOLO                  ")
    print("=========================================================================================")
    compile_all()

    # Cetak laporan akhir status semua proyek
    print("\n=========================================================================================")
    print("                        LAPORAN AUDIT 11 DATASET ROBOFLOW                                ")
    print("=========================================================================================")
    print(f"{'No':<3} | {'Nama Project':<30} | {'Kategori':<10} | {'Status':<15} | {'Keterangan'}")
    print("-" * 89)
    for r in results_table:
        print(f"{r[0]:<3} | {r[1][:30]:<30} | {r[2]:<10} | {r[3]:<15} | {r[4]}")
    print("=========================================================================================")

if __name__ == "__main__":
    main()
