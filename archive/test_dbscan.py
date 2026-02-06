import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def load_las(filepath):
    """
    Charge les données d'un fichier LAS et renvoie les coordonnées et les labels.
    """
    import laspy
    las = laspy.read(filepath)
    coords = np.vstack((las.x, las.y, las.z)).T
    labels = las.classification
    return coords, labels

def visualize_dbscan_clusters(gt_coords, gt_labels, class_id=7, eps=3, min_samples=100):
    """
    Visualiser les clusters créés par DBSCAN pour la classe spécifiée.

    :param gt_coords: Coordonnées des points du GT (Nx3).
    :param gt_labels: Labels de classe des points du GT.
    :param class_id: Identifiant de la classe à analyser.
    :param eps: Paramètre epsilon pour DBSCAN (distance maximale entre deux voisins).
    :param min_samples: Paramètre min_samples pour DBSCAN (nombre minimum de voisins).
    """
    # Extraire les points de la classe spécifiée (par défaut la classe 7)
    class_mask = gt_labels == class_id
    class_coords = gt_coords[class_mask]

    # Appliquer DBSCAN sur les points de la classe spécifiée
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(class_coords)

    # Visualisation des clusters DBSCAN pour la classe
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)

    # Colorier chaque cluster avec une couleur différente
    for label in unique_labels:
        if label == -1:
            color = 'gray'  # Points considérés comme du bruit (outliers)
            label_name = 'Noise'
        else:
            color = plt.cm.jet(label / len(unique_labels))  # Assigner une couleur unique
            label_name = f'Cluster {label}'

        # Afficher les points du cluster avec la couleur choisie
        plt.scatter(class_coords[labels == label, 0],  # Coordonnée x
                    class_coords[labels == label, 1],  # Coordonnée y
                    color=color, label=label_name, alpha=0.6)

    # Ajouter un titre et des étiquettes aux axes
    plt.title(f"DBSCAN Clusters pour la classe {class_id}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger un fichier LAS (exemple)
    gt_coords, gt_labels = load_las("/mnt/d/Eclair_preprod/v13/test_preds/35478_104671_opt.laz")  # Remplace par ton fichier LAS

    # Visualiser les clusters pour la classe 7
    visualize_dbscan_clusters(gt_coords[::4], gt_labels[::4], class_id=7
                              , eps=1.2, min_samples=60)
