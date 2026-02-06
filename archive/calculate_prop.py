import os
import laspy
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm

def compute_class_distribution(directory):
    all_counts = Counter()
    total_points = 0

    point_files = list(Path(directory).rglob("*.laz")) + list(Path(directory).rglob("*.las"))

    for file in tqdm(point_files, desc="Processing point clouds"):
        las = laspy.read(file)
        classes = las.classification


        counts = Counter(classes)
        all_counts.update(counts)
        total_points += len(classes)

    print("\n📊 Classement global :")
    for class_id, count in sorted(all_counts.items()):
        proportion = count / total_points * 100
        print(f"Classe {class_id}: {count} points ({proportion:.2f}%)")


    print(f"\nNombre total de points: {total_points}")

# 🔁 Remplace par ton dossier
if __name__ == "__main__":
    dataset_dir = "data/pointclouds_firstmodel/"  # ⬅️ Mets ton path ici
    compute_class_distribution(dataset_dir)
