"""
Skrip Pengecekan Kelayakan Dataset Roboflow Universe
---------------------------------------------------
Mengecek status ketersediaan, tipe task (Instance Segmentation vs Bounding Box),
jumlah gambar, dan daftar kelas dari URL Roboflow Universe.
"""

import os
import sys
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH)

URLS = [
    "https://universe.roboflow.com/lane1234/lane-k8801",
    "https://universe.roboflow.com/university-of-canberra-its-project/video-data-collection-and-curation-for-two-wheeler-vehicles",
    "https://universe.roboflow.com/road-segmentation/road_segmentationv1.1",
    "https://universe.roboflow.com/curbs/curb-jk8ge",
    "https://universe.roboflow.com/vspace/curb-weau3",
    "https://universe.roboflow.com/rad-khosh-kqqsc/curb-sizeall",
    "https://universe.roboflow.com/autonomousmobilerobotics/road-segmentation-eww07",
    "https://universe.roboflow.com/cvml/road-segmentation-rqcuw",
    "https://universe.roboflow.com/car-racing-f4xxz/road-segmentation-q6nmf",
    "https://universe.roboflow.com/workspace-aabqj/road-segmentation-ipcbf",
    "https://universe.roboflow.com/internship-tvkfo/road-lines-segmentation",
    "https://universe.roboflow.com/goigaxeurc/urc-road-segmentation",
]

def parse_url(url: str):
    clean = url.strip().rstrip('/')
    parts = clean.split('/')
    if len(parts) >= 5 and "universe.roboflow.com" in clean:
        return parts[-2], parts[-1]
    return None, None

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

    print("=========================================================================================")
    print("                 HASIL AUDIT DATASET ROBOFLOW UNIVERSE UNTUK YOLOV8-SEG                  ")
    print("=========================================================================================")
    print(f"{'No':<3} | {'Project ID':<32} | {'Tipe Task':<16} | {'Gambar':<7} | {'Kesesuaian & Catatan'}")
    print("-" * 89)

    rf = Roboflow(api_key=api_key)

    for idx, url in enumerate(URLS, 1):
        ws_name, proj_name = parse_url(url)
        if not ws_name or not proj_name:
            print(f"{idx:<3} | {url:<32} | {'INVALID URL':<16} | {'-':<7} | Format URL tidak valid")
            continue

        try:
            ws = rf.workspace(ws_name)
            proj = ws.project(proj_name)
            p_type = getattr(proj, 'type', 'unknown')
            
            # Coba ambil info versi terbaru
            num_images = "N/A"
            classes = []
            try:
                vers = proj.versions() if callable(getattr(proj, 'versions', None)) else []
                if vers:
                    latest = vers[-1]
                    num_images = str(getattr(latest, 'images', 'N/A'))
            except Exception:
                pass
            except Exception:
                pass

            # Evaluasi kecocokan dengan YOLOv8-seg (butuh Instance Segmentation)
            is_seg = "segmentation" in p_type.lower() or "instance" in p_type.lower()
            
            if is_seg:
                rekomendasi = "COCOK (Instance Segmentation)"
            else:
                rekomendasi = f"TIDAK COCOK ({p_type} - bukan poligon)"

            print(f"{idx:<3} | {proj_name[:32]:<32} | {p_type:<16} | {num_images:<7} | {rekomendasi}")
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "NotFound" in err_msg or "not found" in err_msg.lower():
                status = "TIDAK DITEMUKAN / PRIVATE (404)"
            elif "403" in err_msg or "401" in err_msg:
                status = "AKSES DITOLAK (Private Project)"
            else:
                status = f"ERROR: {err_msg[:30]}"
            print(f"{idx:<3} | {proj_name[:32]:<32} | {'-':<16} | {'-':<7} | {status}")

    print("=========================================================================================")

if __name__ == "__main__":
    main()

