"""Provides a class to apply a postprocessing on vertical objects after a model prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omegaconf import DictConfig


@dataclass
class PLOPostProcessing:
    """Performs a PLO specific post-processing step after classifying a point cloud with Eclair.

    It smartly merges the predictions from the base model and the PLO model

    :param classes: A dictionary that at least maps the classes of `input_classes` to their corresponding
        identifiers in the point cloud.
    :param eps: The epsilon value of the DBSCan, used as a maximum distance between
        points to identify clusters.
    :param min_samples: The minimum number of samples to use for DBScan.
    """

    classes: Mapping[str, int]
    eps: float = 0.03
    min_samples: int = 5

    @staticmethod
    def input_classes() -> set[str]:
        """Classes used during the postprocessing: they must be specified at instantiation in the `classes` mapping."""
        return {
            "bollard",
            "building",
            "pole",
            "sign",
            "trunk",
            "unclassified",
            "vertical_object",
        }

    def __post_init__(self) -> None:
        """Perform a data validation step."""
        missing_classes = self.input_classes() - set(self.classes)
        if missing_classes:
            msg = f"The following classes are missing in the provided mapping: {missing_classes}"
            raise KeyError(msg)

    @classmethod
    def from_config(cls, config: DictConfig) -> Self:
        """Load the post-processing parameters from the configuration of Eclair (usually loaded from a YAML file)."""
        return cls(config.classes, eps=config.dbscan.eps, min_samples=config.dbscan.min_samples)

    def execute(
        self,
        full_pred_base: npt.NDArray[np.uint8],
        full_pred_plo: npt.NDArray[np.uint8],
        coords: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.uint8]:
        """
        Applique un post-traitement et une fusion des prédictions issues de deux modèles.

        - full_pred_base : modèle de base (classification globale)
        - full_pred_plo : modèle PLO (objets verticaux)

        Étapes principales :
        1. Clustering des objets verticaux pour corriger les classes PLO poteau, potelet, panneau.
        2. Fusion des résultats entre les deux modèles selon des règles métier.
        3. Re-clustering pour homogénéiser les prédictions sur les objets verticaux.
        4. Nettoyage et remappage final des classes.

        Retourne : prédiction fusionnée, avec classes remappées.

        :param full_pred_base: Prédictions du premier modèle (shape N).
        :param full_pred_plo: Prédictions du modèle PLO (shape N).
        :param coords: Coordonnées des points (shape N x 3).
        :return: Prédiction fusionnée.
        """
        fused_pred = full_pred_base.copy()

        # I. Clustering des objets verticaux PLO
        class_names = ["pole", "sign", "bollard"]
        vertical_mask = np.isin(full_pred_plo, [self.classes[class_name] for class_name in class_names])
        if not np.any(vertical_mask):
            return fused_pred

        vertical_coords = coords[vertical_mask]
        vertical_labels = full_pred_plo[vertical_mask]
        vertical_indices = np.flatnonzero(vertical_mask)

        cluster_ids = self._cluster_vertical_objects(vertical_coords)

        for cluster_id in np.unique(cluster_ids):
            if cluster_id == -1:
                continue
            cluster_indices = vertical_indices[cluster_ids == cluster_id]

            # Harmonisation potelet/poteau dans le même cluster + comparaison avec classes tronc et batiment
            cluster_labels = vertical_labels[cluster_ids == cluster_id]
            self._fix_pred_2_cluster(full_pred_plo, cluster_indices, cluster_labels)
            if self._is_fused_pred_fix_needed(set(full_pred_base[cluster_indices])):
                self._fix_fused_pred_cluster(fused_pred, cluster_indices, cluster_labels)

        # II. Ajout des potelets. Les potelets sont très souvent raté dans le modèle de base en tant que PLO
        # donc on les rajoute quoi qu'il arrive.
        fused_pred[full_pred_plo == self.classes["bollard"]] = self.classes["bollard"]

        # III. Decision finale sur les classes PLO, poteau et potelet
        fused_pred = self._final_fusion(fused_pred, coords)

        # IV. Nettoyage : suppression des PLO restants qui ne contenaient ni poteau ni potelet
        fused_pred[fused_pred == self.classes["vertical_object"]] = self.classes["unclassified"]
        return fused_pred

    def _cluster_vertical_objects(self, coords: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """
        Applique DBSCAN sur les coordonnées des objets verticaux pour former des clusters.

        :param coords: Coordonnées (N x 3).

        :Return: Tableau d'ID de cluster pour chaque point (-1 = bruit).
        """
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coords[:, :2])

        return cast("npt.NDArray[np.int32]", clustering.labels_)

    def _is_fused_pred_fix_needed(self, base_set: set[int]) -> bool:
        # Si tronc et pas PLO prédit dans le model de base, on ne fait rien
        if self.classes["trunk"] in base_set and self.classes["vertical_object"] not in base_set:
            return False
        # Si cluster ne contient que du bâtiment dans la model de base, on ne fait rien
        return base_set != {self.classes["building"]}

    def _fix_pred_2_cluster(
        self,
        full_pred_plo: npt.NDArray[np.uint8],
        cluster_indices: npt.NDArray[np.int32],
        cluster_labels: npt.NDArray[np.uint8],
    ) -> None:
        """
        Applique des corrections sur les prédictions d'un cluster selon des règles métier.

        Cette fonction modifie les labels d'un cluster de points en appliquant une
        harmonisation potelet/poteau en fonction de la présence d'un panneau ou de la majorité.

        :param full_pred_plo: Prédictions modele PLO à corriger selon les règles.
        :param cluster_indices: Indices des points du cluster dans le nuage complet.
        :param cluster_labels: Labels (à corriger) associés aux points du cluster.
        """
        if self.classes["sign"] in cluster_labels:
            bollard_mask = cluster_labels == self.classes["bollard"]
            cluster_labels[bollard_mask] = self.classes["pole"]
            full_pred_plo[cluster_indices[bollard_mask]] = self.classes["pole"]
        else:  # Si poteau et/ou potelet dans le cluster
            pole_count = np.count_nonzero(cluster_labels == self.classes["pole"])
            bollard_count = np.count_nonzero(cluster_labels == self.classes["bollard"])
            new_label = self.classes["pole"] if pole_count > bollard_count else self.classes["bollard"]
            cluster_labels[:] = new_label
            full_pred_plo[cluster_indices] = new_label

    def _fix_fused_pred_cluster(
        self,
        fused_pred: npt.NDArray[np.uint8],
        cluster_indices: npt.NDArray[np.int32],
        cluster_labels: npt.NDArray[np.uint8],
    ) -> None:
        """
        Applique des corrections sur les prédictions d un cluster selon des règles métier.

        Cette fonction modifie les labels d'un cluster de points en appliquant une
        mise à jour conditionnelle de `fused_pred` pour les classes poteau et panneau.

        :param fused_pred: Prédictions fusionnées, potentiellement modifiées si certaines conditions sont remplies.
        :param cluster_indices: Indices des points du cluster dans le nuage complet.
        :param cluster_labels: Labels associés aux points du cluster.
        """
        for class_name in ("pole", "sign"):
            class_idx = self.classes[class_name]
            fused_pred[cluster_indices[cluster_labels == class_idx]] = class_idx

    def _final_fusion(
        self,
        fused_pred: npt.NDArray[np.uint8],
        coords: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.uint8]:
        """
        Applique un clustering final pour homogénéiser les classes verticales après fusion.

        On complète les prédictions poteau et potelet avec les predictions PLO du premier modele.
            -DBscan sur les classes PLO , poteau et potelet
            -Si un cluster contient PLO et poteau -> On met tout à poteau
            -Si un cluster contient PLO et potelet -> On met tout à potelet


        :param fused_pred: Prédiction fusionnée actuelle.
        :param coords: Coordonnées des points.

        :return: Prédiction mise à jour après homogénéisation.
        """
        target_class_names = ["vertical_object", "bollard", "pole"]
        target_labels = [self.classes[class_name] for class_name in target_class_names]
        mask = np.isin(fused_pred, target_labels)
        if not np.any(mask):
            return fused_pred

        target_coords = coords[mask]
        target_indices = np.flatnonzero(mask)

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(target_coords[:, :2])
        cluster_ids = clustering.labels_

        for cluster_id in np.unique(cluster_ids):
            if cluster_id == -1:
                continue
            cluster_indices = target_indices[cluster_ids == cluster_id]
            cluster_labels = fused_pred[cluster_indices]

            if self.classes["vertical_object"] in cluster_labels:
                if self.classes["pole"] in cluster_labels:
                    fused_pred[cluster_indices] = self.classes["pole"]
                elif self.classes["bollard"] in cluster_labels:
                    fused_pred[cluster_indices] = self.classes["bollard"]
        return fused_pred
