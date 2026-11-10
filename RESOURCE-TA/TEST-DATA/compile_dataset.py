"""
Skrip Kompilator Dataset Mandiri (Dataset Builder & Compiler)
-------------------------------------------------------------
Memindai seluruh subfolder di dalam folder 'Dataset/':
  - Dataset/Curb/     -> Seluruh label trotoar dipetakan ke Kelas 0 (curb)
  - Dataset/Road/     -> Seluruh label aspal jalan dipetakan ke Kelas 1 (road)
  - Dataset/Line/     -> Seluruh label marka jalan dipetakan ke Kelas 1 (road)
  - Dataset/Obstacle/ -> (Opsional / Masa depan)
  - Data Asli BeamNG  -> (800 citra dasar tersimpan aman)

Menghasilkan folder siap-latih 'dataset_yolo/' lengkap dengan 'data.yaml'.
"""

import os
import glob
import yaml
import shutil
import argparse

BASE_DIR = os.path.dirname(__file__)
DATASET_ROOT = os.path.join(BASE_DIR, "Dataset")
TARGET_YOLO = os.path.join(BASE_DIR, "dataset_yolo")
RAW_BEAMNG = os.path.join(BASE_DIR, "dataset_raw")

def get_class_mapping(data_yaml_path, default_class=1):
    mapping = {}
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                d = yaml.safe_load(f)
            names = d.get('names', [])
            if isinstance(names, list):
                names_dict = {i: name for i, name in enumerate(names)}
            elif isinstance(names, dict):
                names_dict = {int(k): v for k, v in names.items()}
            else:
                names_dict = {}

            for orig_id, name in names_dict.items():
                nl = str(name).lower()
                if any(k in nl for k in ['curb', 'sidewalk', 'pavement']):
                    mapping[orig_id] = 0
                elif any(k in nl for k in ['road', 'street', 'asphalt', 'lane', 'line', 'driveway']):
                    mapping[orig_id] = 1
                else:
                    mapping[orig_id] = None
        except Exception:
            pass

    if not mapping:
        mapping = {0: default_class, 1: 1 if default_class == 0 else 0}
    return mapping

def compile_all():
    print("=========================================================================================")
    print("                    DATASET COMPILER & BUILDER UNTUK YOLOV8-SEG                          ")
    print("=========================================================================================")
    print(f"Folder Sumber Dataset : {DATASET_ROOT}")
    print(f"Folder Target Pelatihan: {TARGET_YOLO}")
    print("-----------------------------------------------------------------------------------------")

    for split in ['train', 'val']:
        os.makedirs(os.path.join(TARGET_YOLO, "images", split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_YOLO, "labels", split), exist_ok=True)

    category_stats = {}

    # 1. Hitung data dasar BeamNG yang sudah ada
    beamng_train = len(glob.glob(os.path.join(TARGET_YOLO, "images", "train", "frame_*.jpg")))
    beamng_val = len(glob.glob(os.path.join(TARGET_YOLO, "images", "val", "frame_*.jpg")))
    category_stats["BeamNG (Simulator Asli)"] = beamng_train + beamng_val

    # 2. Kategori yang akan dipindai di folder Dataset/
    categories = [
        ("Curb", 0, "rf_curb_"),
        ("Road", 1, "rf_road_"),
        ("Line", 1, "rf_line_"),
    ]

    total_added = 0

    for cat_name, default_cls, prefix in categories:
        cat_dir = os.path.join(DATASET_ROOT, cat_name)
        if not os.path.exists(cat_dir):
            category_stats[cat_name] = 0
            continue

        cat_count = 0

        # Cari seluruh subproject di dalam kategori ini
        sub_items = [os.path.join(cat_dir, d) for d in os.listdir(cat_dir)]
        subprojects = [d for d in sub_items if os.path.isdir(d)]
        if not subprojects:
            subprojects = [cat_dir]  # Jika gambar diletakkan langsung di dalam folder kategori

        for proj_dir in subprojects:
            proj_name = os.path.basename(proj_dir)
            yaml_path = os.path.join(proj_dir, "data.yaml")
            class_map = get_class_mapping(yaml_path, default_class=default_cls)

            # Cek split Roboflow atau direktori standar
            splits_map = {
                'train': ('train', 'train'),
                'valid': ('val', 'valid'),
                'test':  ('val', 'test')
            }

            for src_split, (tgt_split, _) in splits_map.items():
                src_img_dir = os.path.join(proj_dir, src_split, "images")
                src_lbl_dir = os.path.join(proj_dir, src_split, "labels")

                # Cek alternatif jika tidak ada subfolder train/valid/test
                if not os.path.exists(src_img_dir):
                    if src_split == 'train':
                        src_img_dir = os.path.join(proj_dir, "images")
                        src_lbl_dir = os.path.join(proj_dir, "labels")
                        if not os.path.exists(src_img_dir):
                            src_img_dir = proj_dir
                            src_lbl_dir = proj_dir

                if not os.path.exists(src_img_dir):
                    continue

                tgt_img_dir = os.path.join(TARGET_YOLO, "images", tgt_split)
                tgt_lbl_dir = os.path.join(TARGET_YOLO, "labels", tgt_split)

                img_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    img_files.extend(glob.glob(os.path.join(src_img_dir, ext)))

                for img_path in img_files:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]

                    lbl_path = os.path.join(src_lbl_dir, f"{base_name}.txt")

                    safe_prefix = f"{prefix}{proj_name[:8]}_"
                    new_base = f"{safe_prefix}{base_name}"
                    new_img_path = os.path.join(tgt_img_dir, f"{new_base}{os.path.splitext(img_path)[1]}")
                    new_lbl_path = os.path.join(tgt_lbl_dir, f"{new_base}.txt")

                    shutil.copy2(img_path, new_img_path)

                    if os.path.exists(lbl_path):
                        with open(lbl_path, 'r', encoding='utf-8') as f_in, open(new_lbl_path, 'w', encoding='utf-8') as f_out:
                            for line in f_in:
                                parts = line.strip().split()
                                if not parts:
                                    continue
                                try:
                                    orig_id = int(parts[0])
                                except ValueError:
                                    continue
                                # Hanya simpan baris poligon murni (minimal 3 titik = 6 koordinat -> total 7 elemen)
                                # Membuang baris bounding-box kotak (5 elemen) yang merusak segmentasi YOLOv8
                                if len(parts) >= 7 and len(parts) % 2 == 1:
                                    mapped_id = class_map.get(orig_id, default_cls)
                                    if mapped_id is not None:
                                        parts[0] = str(mapped_id)
                                        f_out.write(" ".join(parts) + "\n")
                    else:
                        open(new_lbl_path, 'a').close()

                    cat_count += 1
                    total_added += 1

        category_stats[f"Folder: {cat_name}"] = cat_count

    # 3. Hapus cache ultralytics agar cache lama ter-refresh
    for c in glob.glob(os.path.join(TARGET_YOLO, "labels", "*.cache")):
        try:
            os.remove(c)
        except Exception:
            pass

    # 4. Tulis ulang data.yaml terpadu
    yaml_content = f"""# Konfigurasi Dataset YOLOv8-seg Terpadu (BeamNG + Kategori Eksternal)
path: {os.path.abspath(TARGET_YOLO).replace(os.sep, '/')}
train: images/train
val: images/val

names:
  0: curb
  1: road
"""
    with open(os.path.join(TARGET_YOLO, "data.yaml"), 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # 4. Ringkasan
    n_train = len(glob.glob(os.path.join(TARGET_YOLO, "images", "train", "*.*")))
    n_val = len(glob.glob(os.path.join(TARGET_YOLO, "images", "val", "*.*")))

    print(f"{'Kategori / Sumber Data':<35} | {'Jumlah Citra':<15}")
    print("-" * 55)
    for cat, count in category_stats.items():
        print(f"{cat:<35} | {count:<15}")
    print("=" * 55)
    print(f"Total Citra Train     : {n_train} citra")
    print(f"Total Citra Validasi  : {n_val} citra")
    print(f"TOTAL DATASET SIAP    : {n_train + n_val} CITRA")
    print("=========================================================================================")
    print("\nDataset siap digunakan untuk pelatihan!")
    print('Perintah latihan: "C:\\Users\\Noapllabib\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" train_yolo_seg.py\n')

if __name__ == "__main__":
    compile_all()
