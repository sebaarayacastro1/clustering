"""Préparation, estimation mémoire et classification ascendante hiérarchique."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster


def charger_coordonnees_et_identifiants(racine: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Charge les sorties de l'ACM final, dans l'ordre exact de l'échantillon."""
    racine = Path(racine)
    chemin_coordonnees = racine / "resultats/coordonnees_factorielles.csv"
    chemin_correspondance = racine / "resultats/correspondance_clients_acm.csv"
    chemin_metadonnees = racine / "modeles/acm_final_metadonnees.json"
    for chemin in (chemin_coordonnees, chemin_correspondance, chemin_metadonnees):
        if not chemin.is_file():
            raise FileNotFoundError(f"Fichier manquant : {chemin}. Exécuter d'abord 02_acm.ipynb.")
    coordonnees = pd.read_csv(chemin_coordonnees)
    identifiants = pd.read_csv(chemin_correspondance)
    with chemin_metadonnees.open(encoding="utf-8") as fichier:
        metadonnees = json.load(fichier)
    if coordonnees.empty or coordonnees.shape[1] < 1:
        raise ValueError("Les coordonnées factorielles sont vides.")
    if len(coordonnees) != len(identifiants):
        raise ValueError("Le nombre d'identifiants diffère du nombre de coordonnées.")
    positions = np.arange(len(coordonnees))
    if "position_echantillon" not in identifiants.columns or not np.array_equal(
        identifiants["position_echantillon"].to_numpy(), positions
    ):
        raise ValueError("L'ordre des identifiants ne correspond pas aux coordonnées.")
    if metadonnees.get("iteration") != 3:
        raise ValueError("Les métadonnées ne désignent pas l'ACM final (itération 3).")
    if int(metadonnees.get("nombre_axes_calcules", -1)) != coordonnees.shape[1]:
        raise ValueError("Le nombre d'axes ne correspond pas aux métadonnées.")
    if not np.isfinite(coordonnees.to_numpy(dtype=float)).all():
        raise ValueError("Les coordonnées contiennent des valeurs non finies.")
    return coordonnees, identifiants, metadonnees


def choisir_axes(coordonnees: pd.DataFrame, nombre_axes: int) -> pd.DataFrame:
    """Conserve les premiers axes, sans modifier le modèle ACM ajusté."""
    if not 1 <= nombre_axes <= coordonnees.shape[1]:
        raise ValueError("Nombre d'axes invalide.")
    donnees = coordonnees.iloc[:, :nombre_axes].copy()
    if donnees.isna().any().any():
        raise ValueError("Valeurs manquantes dans les coordonnées.")
    return donnees


def selectionner_sous_echantillon(
    nombre_individus: int, taille: int | None, graine: int = 42
) -> np.ndarray:
    """Retourne les positions, triées, pour préserver la correspondance des lignes."""
    if nombre_individus < 2:
        raise ValueError("Il faut au moins deux individus.")
    if taille is None or taille >= nombre_individus:
        return np.arange(nombre_individus, dtype=int)
    if taille < 2:
        raise ValueError("La taille du sous-échantillon doit être supérieure à 1.")
    return np.sort(np.random.default_rng(graine).choice(nombre_individus, size=taille, replace=False))


def estimer_memoire_distances(nombre_individus: int) -> dict:
    """Estimation minimale du tableau condensé en float64 (sans les copies et surcoûts)."""
    if nombre_individus < 2:
        raise ValueError("Il faut au moins deux individus.")
    nombre_paires = nombre_individus * (nombre_individus - 1) // 2
    octets = nombre_paires * np.dtype("float64").itemsize
    return {"individus": nombre_individus, "paires": nombre_paires,
            "memoire_minimale_gio": octets / 1024**3,
            "memoire_prudente_gio": 2.5 * octets / 1024**3}


def effectuer_cah(
    coordonnees: pd.DataFrame, budget_memoire_gio: float = 8.0,
    confirmation_depassement: bool = False, moteur: str = "scipy"
) -> np.ndarray:
    """Calcule Ward en distance euclidienne avec garde-fou mémoire explicite."""
    if len(coordonnees) < 3:
        raise ValueError("La CAH requiert au moins trois individus.")
    if budget_memoire_gio <= 0:
        raise ValueError("Budget mémoire invalide.")
    valeurs = np.asarray(coordonnees, dtype=np.float64)
    if not np.isfinite(valeurs).all():
        raise ValueError("Coordonnées non finies.")
    estimation = estimer_memoire_distances(len(valeurs))
    if (estimation["memoire_prudente_gio"] > budget_memoire_gio
            and not confirmation_depassement):
        raise MemoryError(
            "CAH interrompue : mémoire estimée à environ "
            f"{estimation['memoire_prudente_gio']:.1f} Gio, supérieure au "
            f"budget de {budget_memoire_gio:.1f} Gio. "
            "Réduire la sous-population ou confirmer explicitement après "
            "vérification des ressources de la machine."
        )
    if moteur == "scipy":
        return linkage(valeurs, method="ward", metric="euclidean", optimal_ordering=False)
    if moteur == "fastcluster":
        try:
            import fastcluster
        except ImportError as erreur:
            raise ImportError("Installer fastcluster ou sélectionner le moteur 'scipy'.") from erreur
        return fastcluster.linkage_vector(valeurs, method="ward", metric="euclidean")
    raise ValueError("Moteur inconnu : choisir 'scipy' ou 'fastcluster'.")


def obtenir_partition(hierarchie: np.ndarray, nombre_groupes: int) -> np.ndarray:
    """Produit exactement k groupes en coupant l'arbre de Ward."""
    nombre_individus = hierarchie.shape[0] + 1
    if not 2 <= nombre_groupes <= nombre_individus:
        raise ValueError("Nombre de groupes invalide.")
    # cut_tree assure exactement k groupes (fcluster(maxclust) peut retourner moins).
    from scipy.cluster.hierarchy import cut_tree
    return cut_tree(hierarchie, n_clusters=[nombre_groupes]).ravel().astype(int) + 1


def tailles_groupes(etiquettes: np.ndarray, scenario: str) -> pd.DataFrame:
    """Décrit les effectifs et proportions d'une partition."""
    serie = pd.Series(etiquettes, name="Groupe")
    tableau = serie.value_counts().sort_index().rename_axis("Groupe").reset_index(name="Effectif")
    tableau.insert(0, "Scénario", scenario)
    tableau["Part (%)"] = 100 * tableau["Effectif"] / tableau["Effectif"].sum()
    return tableau


def hauteurs_fusions(hierarchie: np.ndarray, nombre_dernieres: int = 30) -> pd.DataFrame:
    """Extrait les hauteurs des dernières fusions, pour examiner les sauts."""
    hauteurs = np.asarray(hierarchie[:, 2], dtype=float)
    debut = max(0, len(hauteurs) - nombre_dernieres)
    valeurs = hauteurs[debut:]
    sauts = np.diff(hauteurs, prepend=np.nan)[debut:]
    return pd.DataFrame({
        "Rang_fusion": np.arange(debut + 1, len(hauteurs) + 1),
        "Hauteur": valeurs,
        "Saut_hauteur": sauts,
    })
