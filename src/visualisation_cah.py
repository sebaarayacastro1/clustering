"""Graphiques de diagnostic pour la classification hiérarchique."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def afficher_dendrogramme(hierarchie: np.ndarray, derniere_fusions: int = 35) -> None:
    """Dendrogramme tronqué pour garder un affichage lisible."""
    fig, axe = plt.subplots(figsize=(12, 5))
    dendrogram(hierarchie, truncate_mode="lastp", p=derniere_fusions,
               show_leaf_counts=True, leaf_rotation=65, ax=axe)
    axe.set(title="Dendrogramme de Ward (dernières fusions)",
            xlabel="Groupes agrégés", ylabel="Hauteur de fusion")
    axe.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()


def afficher_hauteurs(hierarchie: np.ndarray, derniere_fusions: int = 30) -> None:
    """Visualisation des hauteurs de fusion et de leurs ruptures."""
    hauteurs = hierarchie[:, 2]
    debut = max(0, len(hauteurs) - derniere_fusions)
    fig, axe = plt.subplots(figsize=(10, 4))
    axe.plot(np.arange(debut + 1, len(hauteurs) + 1), hauteurs[debut:], marker="o")
    axe.set(title="Hauteurs des dernières fusions", xlabel="Rang de la fusion",
            ylabel="Hauteur de Ward")
    axe.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
