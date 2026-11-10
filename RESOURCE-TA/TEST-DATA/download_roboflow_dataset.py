"""
Skrip Pengunduh Dataset Jalan & Trotoar dari Roboflow Universe
--------------------------------------------------------------
Menggunakan Roboflow Python SDK dan API Key Akun Kampus.
Mendukung pengunduhan langsung dalam format YOLOv8 Segmentation.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Muat kredensial dari .env tanpa mencetak nilai ke terminal
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH)

def parse_universe_url(url: str):
    """
    Mengekstrak workspace dan project name dari URL Roboflow Universe.
    Contoh: https://universe.roboflow.com/workspace-name/project-name
    """
    clean_url = url.strip().rstrip('/')
    parts = clean_url.split('/')
    if len(parts) >= 5 and "universe.roboflow.com" in clean_url:
        workspace = parts[-2]
        project = parts[-1]
        return workspace, project
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Download Dataset Jalan/Trotoar dari Roboflow Universe")
    parser.add_argument("--url", type=str, default="",
                        help="URL lengkap halaman dataset di Roboflow Universe (contoh: https://universe.roboflow.com/...)")
    parser.add_argument("--workspace", type=str, default="",
                        help="Nama workspace di Roboflow")
    parser.add_argument("--project", type=str, default="",
                        help="Nama project dataset di Roboflow")
    parser.add_argument("--version", type=int, default=1,
                        help="Nomor versi dataset (default: 1)")
    parser.add_argument("--format", type=str, default="yolov8",
                        help="Format ekspor (default: yolov8)")
    parser.add_argument("--category", type=str, default="",
                        choices=["", "Road", "Curb", "Line", "Sign", "Obstacle"],
                        help="Kategori folder penyimpanan di dalam 'Dataset/' (contoh: Road, Curb, Line)")
    parser.add_argument("--target_dir", type=str, default="dataset_external_roboflow",
                        help="Folder tujuan penyimpanan dataset jika --category tidak diisi")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("\n[ERROR] Kredensial ROBOFLOW_API_KEY tidak ditemukan!")
        print("Panduan:")
        print("1. Buka https://app.roboflow.com (login akun kampus)")
        print("2. Masuk ke Settings -> Workspaces -> Roboflow API -> Private API Key")
        print("3. Simpan ke berkas .env menggunakan perintah aman di terminal.")
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n[ERROR] Paket 'roboflow' belum terpasang.")
        print("Jalankan perintah ini di PowerShell:")
        print('  & "C:\\Users\\Noapllabib\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" -m pip install roboflow')
        return

    workspace = args.workspace
    project = args.project

    if args.url:
        w_url, p_url = parse_universe_url(args.url)
        if w_url and p_url:
            workspace = w_url
            project = p_url

    if not workspace or not project:
        print("\n[INFO] Masukkan nama workspace dan project Roboflow.")
        print("Contoh penggunaan:")
        print('  python download_roboflow_dataset.py --workspace "mohamed-traore-2wbn5" --project "curb-detection" --version 1')
        print('  python download_roboflow_dataset.py --url "https://universe.roboflow.com/mohamed-traore-2wbn5/curb-detection" --version 1')
        return

    print("===========================================================")
    print("      ROBOFLOW UNIVERSE DATASET DOWNLOADER                 ")
    print("===========================================================")
    print(f"-> Workspace : {workspace}")
    print(f"-> Project   : {project}")
    print(f"-> Versi     : {args.version}")
    print(f"-> Format    : {args.format}")
    print(f"-> Target    : {args.target_dir}")
    print("-----------------------------------------------------------")

    try:
        rf = Roboflow(api_key=api_key)
        ws = rf.workspace(workspace)
        proj = ws.project(project)
        ver = proj.version(args.version)

        if args.category:
            out_path = os.path.join(os.path.dirname(__file__), "Dataset", args.category, project)
        else:
            out_path = os.path.join(os.path.dirname(__file__), args.target_dir)

        print("-> Memulai proses pengunduhan dataset...")
        dataset = ver.download(args.format, location=out_path, overwrite=True)
        print(f"\n[SUKSES] Dataset berhasil diunduh ke: {dataset.location}")
        print("File data.yaml dan gambar siap digunakan untuk pelatihan YOLOv8-seg!")
    except Exception as e:
        print(f"\n[GAGAL] Terjadi kesalahan saat mengunduh dataset: {e}")

if __name__ == "__main__":
    main()

