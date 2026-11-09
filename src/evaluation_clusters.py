"""Indicateurs exploratoires et comparaisons des partitions de CAH."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score


def calculer_silhouette_echantillon(
    coordonnees: pd.DataFrame, etiquettes: np.ndarray,
    taille_maximale: int = 1200, graine: int = 42
) -> float:
    """Silhouette approximative sur un sous-échantillon, pour limiter la mémoire."""
    if len(coordonnees) != len(etiquettes):
        raise ValueError("Les coordonnées et les étiquettes ne sont pas alignées.")
    if len(np.unique(etiquettes)) < 2:
        return float("nan")
    taille = min(taille_maximale, len(coordonnees))
    if taille < 2:
        return float("nan")
    # Prélever dans chaque groupe si possible pour éviter les groupes absents.
    rng = np.random.default_rng(graine)
    indices = []
    classes, effectifs = np.unique(etiquettes, return_counts=True)
    if taille < len(classes):
        return float("nan")
    for classe in classes:
        membres = np.flatnonzero(etiquettes == classe)
        indices.append(int(rng.choice(membres)))
    autres = np.setdiff1d(np.arange(len(etiquettes)), np.asarray(indices), assume_unique=True)
    if taille > len(indices):
        indices.extend(rng.choice(autres, size=taille - len(indices), replace=False).tolist())
    indices = np.array(indices)
    sous_etiquettes = np.asarray(etiquettes)[indices]
    if len(np.unique(sous_etiquettes)) >= len(indices):
        return float("nan")
    return float(silhouette_score(coordonnees.iloc[indices].to_numpy(), sous_etiquettes))


def comparer_partitions(reference: np.ndarray, autre: np.ndarray, nom_reference: str,
                        nom_autre: str) -> pd.DataFrame:
    """Compare les partitions sur les mêmes individus, indépendamment des numéros de groupes."""
    if len(reference) != len(autre):
        raise ValueError("Les partitions doivent contenir les mêmes individus dans le même ordre.")
    return pd.DataFrame([{"Référence": nom_reference, "Comparaison": nom_autre,
                          "ARI": adjusted_rand_score(reference, autre)}])


def tableau_croise_partitions(reference: np.ndarray, autre: np.ndarray) -> pd.DataFrame:
    """Visualise les regroupements et subdivisions de k vers k ± 1."""
    if len(reference) != len(autre):
        raise ValueError("Les partitions ne sont pas alignées.")
    return pd.crosstab(pd.Series(reference, name="Groupe référence"),
                       pd.Series(autre, name="Groupe comparé"))
