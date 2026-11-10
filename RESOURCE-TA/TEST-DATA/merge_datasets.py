"""
Skrip Penggabung Dataset Eksternal (Roboflow) ke Dataset Pelatihan YOLOv8
-----------------------------------------------------------------------
Dilengkapi dengan 'Smart Class Remapper':
Menganalisis file 'data.yaml' dari Roboflow dan secara otomatis menyelaraskan ID kelas:
  - Kelas Trotoar ('curb', 'sidewalk') -> ID 0
  - Kelas Jalan ('road', 'street', 'asphalt', 'lane', 'line') -> ID 1
Mencegah kerusakan label (Label Corruption) saat menggabungkan berbagai dataset!
"""

import os
import glob
import yaml
import shutil
import argparse

def get_class_remapper(data_yaml_path):
    """
    Mendeteksi pemetaan kelas dari data.yaml sumber ke format target kita:
      Target 0 : curb
      Target 1 : road
    """
    mapping = {}
    if not os.path.exists(data_yaml_path):
        # Default jika data.yaml tidak ada: anggap 0=curb, 1=road
        return {0: 0, 1: 1}

    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        names = data.get('names', [])
        if isinstance(names, list):
            names_dict = {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            names_dict = {int(k): v for k, v in names.items()}
        else:
            names_dict = {0: 'curb', 1: 'road'}

        print("  [INFO] Deteksi Kelas di Sumber:")
        for orig_id, name in names_dict.items():
            name_lower = str(name).lower()
            if any(k in name_lower for k in ['curb', 'sidewalk', 'pavement']):
                mapping[orig_id] = 0
                print(f"    • ID {orig_id} ('{name}') -> Dipetakan ke Kelas 0 (curb)")
            elif any(k in name_lower for k in ['road', 'street', 'asphalt', 'lane', 'line', 'driveway']):
                mapping[orig_id] = 1
                print(f"    • ID {orig_id} ('{name}') -> Dipetakan ke Kelas 1 (road)")
            else:
                mapping[orig_id] = None
                print(f"    • ID {orig_id} ('{name}') -> DIABAIKAN (Bukan trotoar/jalan)")

    except Exception as e:
        print(f"  [WARNING] Gagal membaca data.yaml ({e}), menggunakan pemetaan default 0:0, 1:1")
        mapping = {0: 0, 1: 1}

    return mapping

def merge(source_dir, target_dir, prefix="rf_"):
    print("===========================================================")
    print("      PENGGABUNG DATASET ROBOFLOW (SMART CLASS REMAPPER)   ")
    print("===========================================================")
    print(f"-> Sumber : {source_dir}")
    print(f"-> Target : {target_dir}")
    print(f"-> Prefix : {prefix}")
    print("-" * 59)

    if not os.path.exists(source_dir):
        print(f"Error: Folder sumber '{source_dir}' tidak ditemukan!")
        return 0

    yaml_path = os.path.join(source_dir, "data.yaml")
    class_map = get_class_remapper(yaml_path)

    splits_map = {
        'train': ('train', 'train'),
        'valid': ('val', 'valid'),
        'test':  ('val', 'test')
    }

    total_merged = 0

    for src_split, (tgt_split, _) in splits_map.items():
        src_img_dir = os.path.join(source_dir, src_split, "images")
        src_lbl_dir = os.path.join(source_dir, src_split, "labels")

        tgt_img_dir = os.path.join(target_dir, "images", tgt_split)
        tgt_lbl_dir = os.path.join(target_dir, "labels", tgt_split)

        if not os.path.exists(src_img_dir):
            continue

        os.makedirs(tgt_img_dir, exist_ok=True)
        os.makedirs(tgt_lbl_dir, exist_ok=True)

        images = glob.glob(os.path.join(src_img_dir, "*.*"))
        split_count = 0

        for img_path in images:
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(src_lbl_dir, f"{base_name}.txt")

            new_base = f"{prefix}{base_name}"
            new_img_path = os.path.join(tgt_img_dir, f"{new_base}{ext}")
            new_lbl_path = os.path.join(tgt_lbl_dir, f"{new_base}.txt")

            # Salin gambar
            shutil.copy2(img_path, new_img_path)

            # Proses dan Remap Label Poligon
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

                        mapped_id = class_map.get(orig_id, None)
                        if mapped_id is not None:
                            parts[0] = str(mapped_id)
                            f_out.write(" ".join(parts) + "\n")
            else:
                open(new_lbl_path, 'a').close()

            split_count += 1
            total_merged += 1

        print(f"-> Split '{src_split}' -> Target '{tgt_split}': {split_count} pasang data disalin.")

    n_train = len(glob.glob(os.path.join(target_dir, "images", "train", "*.*")))
    n_val = len(glob.glob(os.path.join(target_dir, "images", "val", "*.*")))

    print("-" * 59)
    print(f"PENGGABUNGAN SUKSES! Total {total_merged} file ditambahkan.")
    print(f"Jumlah Dataset YOLO Sekarang:")
    print(f"  • Train: {n_train} citra")
    print(f"  • Val  : {n_val} citra")
    print(f"  • Total: {n_train + n_val} citra siap latih.")
    print("===========================================================")
    return total_merged

def main():
    parser = argparse.ArgumentParser(description="Gabungkan dataset Roboflow ke dataset_yolo dengan Smart Remapper")
    parser.add_argument("--source", type=str, default="dataset_external_roboflow",
                        help="Folder sumber dataset Roboflow")
    parser.add_argument("--target", type=str, default="dataset_yolo",
                        help="Folder target dataset_yolo")
    parser.add_argument("--prefix", type=str, default="rf_curb_",
                        help="Prefix nama file untuk mencegah bentrok nama")
    args = parser.parse_args()

    merge(args.source, args.target, args.prefix)

if __name__ == "__main__":
    main()
