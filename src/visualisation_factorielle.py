"""Représentation descriptive des partitions CAH dans l'espace de l'ACM."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def charger_coordonnees_partitions(
    racine: str | Path, partitions: pd.DataFrame, nombre_axes: int = 4
) -> pd.DataFrame:
    """Retrouve exactement les coordonnées ACM des individus étiquetés par la CAH.

    Les positions sont celles du fichier correspondance_clients_acm.csv, et non
    les indices du CSV brut. Les identifiants sont vérifiés lorsqu'ils existent.
    """
    racine = Path(racine)
    chemin_coordonnees = racine / "resultats" / "coordonnees_factorielles.csv"
    chemin_identifiants = racine / "resultats" / "correspondance_clients_acm.csv"
    coordonnees = pd.read_csv(chemin_coordonnees)
    correspondance = pd.read_csv(chemin_identifiants)
    if len(coordonnees) != len(correspondance):
        raise ValueError("Coordonnées et correspondance ACM de tailles différentes.")
    if nombre_axes < 4 or coordonnees.shape[1] < nombre_axes:
        raise ValueError("Les quatre premiers axes de l'ACM doivent être disponibles.")
    if "position_echantillon" not in correspondance or "position_echantillon" not in partitions:
        raise ValueError("La position dans l'échantillon ACM est indispensable au tracé.")
    positions_reference = pd.to_numeric(correspondance["position_echantillon"], errors="raise")
    if not np.array_equal(positions_reference.to_numpy(), np.arange(len(correspondance))):
        raise ValueError("L'ordre des positions de l'ACM n'est pas cohérent.")
    positions = pd.to_numeric(partitions["position_echantillon"], errors="raise")
    if (positions.isna().any() or (positions % 1 != 0).any()
            or positions.duplicated().any()):
        raise ValueError("Positions CAH manquantes, répétées ou non entières.")
    indices = positions.to_numpy(dtype=int)
    if ((indices < 0) | (indices >= len(coordonnees))).any():
        raise ValueError("Une position CAH est en dehors de l'échantillon ACM.")
    for identifiant in ("id_anonyme", "indice_ligne_source"):
        if identifiant in partitions.columns and identifiant in correspondance.columns:
            valeurs_cah = partitions[identifiant].reset_index(drop=True).astype("string")
            valeurs_acm = correspondance.iloc[indices][identifiant].reset_index(drop=True).astype("string")
            if not valeurs_cah.equals(valeurs_acm):
                raise ValueError(f"Incohérence des individus pour : {identifiant}.")
    resultat = coordonnees.iloc[indices, :nombre_axes].reset_index(drop=True)
    if not np.isfinite(resultat.to_numpy(dtype=float)).all():
        raise ValueError("Coordonnées factorielles non finies.")
    return resultat


def afficher_partitions_factorielles(
    coordonnees: pd.DataFrame,
    partitions: pd.DataFrame,
    valeurs_k: list[int],
    axes: tuple[int, int] = (1, 2),
    maximum_points: int = 6000,
    graine: int = 42,
    chemin_sortie: str | Path | None = None,
) -> None:
    """Trace des points colorés et centroïdes, sans simuler de frontières.

    Le sous-échantillonnage des points est identique entre partitions ; les
    centroïdes reposent toujours sur la totalité des individus.
    """
    if len(partitions) != len(coordonnees):
        raise ValueError("Coordonnées et étiquettes CAH non alignées.")
    if maximum_points <= 0:
        raise ValueError("maximum_points doit être strictement positif.")
    a, b = axes
    if a == b or min(axes) < 1 or max(axes) > coordonnees.shape[1]:
        raise ValueError("Indices des axes factoriels invalides.")
    valeurs = coordonnees.iloc[:, [a - 1, b - 1]].to_numpy(dtype=float)
    n = len(valeurs)
    indices = (np.sort(np.random.default_rng(graine).choice(n, size=maximum_points, replace=False))
               if n > maximum_points else np.arange(n))
    fig, graphiques = plt.subplots(1, len(valeurs_k), figsize=(5.5 * len(valeurs_k), 5),
                                   sharex=True, sharey=True, squeeze=False)
    bornes_x = np.quantile(valeurs[:, 0], [0.002, 0.998]); bornes_y = np.quantile(valeurs[:, 1], [0.002, 0.998])
    marge_x = max(0.1, (bornes_x[1] - bornes_x[0]) * 0.07)
    marge_y = max(0.1, (bornes_y[1] - bornes_y[0]) * 0.07)
    for graphique, k in zip(graphiques[0], valeurs_k):
        colonne = f"groupe_k{k}"
        if colonne not in partitions:
            raise ValueError(f"Colonne CAH absente : {colonne}.")
        etiquettes = partitions[colonne].to_numpy()
        groupes = sorted(np.unique(etiquettes))
        couleurs = plt.get_cmap("tab10")
        for indice_groupe, groupe in enumerate(groupes):
            masque = etiquettes[indices] == groupe
            couleur = couleurs(indice_groupe % 10)
            graphique.scatter(valeurs[indices[masque], 0], valeurs[indices[masque], 1],
                              s=6, alpha=0.30, color=couleur, edgecolors="none",
                              rasterized=True, label=f"G{groupe}")
            centre = valeurs[etiquettes == groupe].mean(axis=0)
            graphique.scatter([centre[0]], [centre[1]], s=95, marker="X",
                              color=couleur, edgecolor="black", linewidth=0.7, zorder=5)
            graphique.annotate(f"G{groupe}", centre, xytext=(5, 5),
                               textcoords="offset points", fontsize=9, weight="bold")
        graphique.set_title(f"k = {k}")
        graphique.set_xlabel(f"Dimension {a} de l'ACM")
        graphique.set_xlim(bornes_x[0] - marge_x, bornes_x[1] + marge_x)
        graphique.set_ylim(bornes_y[0] - marge_y, bornes_y[1] + marge_y)
        graphique.grid(alpha=0.15)
    graphiques[0][0].set_ylabel(f"Dimension {b} de l'ACM")
    fig.suptitle("Partitions CAH dans l'espace factoriel — points échantillonnés, centres complets", fontsize=11)
    fig.tight_layout()
    if chemin_sortie:
        chemin = Path(chemin_sortie)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin, dpi=170, bbox_inches="tight")
    plt.show()
    plt.close(fig)
