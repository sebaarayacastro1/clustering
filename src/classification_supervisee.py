"""Apprentissage supervisé des étiquettes de la CAH de référence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


NOMS_METIER_K5 = {
    1: "Clients fidèles à forte valeur",
    2: "Clients en churn à faible fidélité",
    3: "Clients sans activité observée",
    4: "Clients intermédiaires souvent réactivés",
    5: "Clients récents en phase d'intégration",
}


def charger_reference_cah(racine: str | Path, *, taille: int = 20000,
                          axes: int = 4, graine: int = 42, k: int = 5):
    """Charge et vérifie la CAH et ses coordonnées, sans réentraîner l'ACM."""
    from src.classification_hierarchique import (
        charger_coordonnees_et_identifiants, selectionner_sous_echantillon
    )

    racine = Path(racine)
    dossier = racine / "resultats" / "cah"
    chemin_partitions = dossier / f"partitions_essai_{axes}axes.csv"
    chemin_parametres = dossier / f"parametres_essai_{axes}axes.json"
    for chemin in (chemin_partitions, chemin_parametres):
        if not chemin.is_file():
            raise FileNotFoundError(f"Fichier de la CAH de référence manquant : {chemin}")
    with chemin_parametres.open(encoding="utf-8") as fichier:
        parametres = json.load(fichier)
    attendus = {"mode": "essai", "nombre_individus": taille, "nombre_axes": axes,
                "k_central": k, "graine": graine, "methode": "ward"}
    for cle, attendu in attendus.items():
        if parametres.get(cle) != attendu:
            raise ValueError(f"Paramètre CAH incorrect : {cle}={parametres.get(cle)!r} (attendu : {attendu!r}).")
    if k not in parametres.get("valeurs_k", []):
        raise ValueError("La partition demandée ne figure pas dans l'expérience CAH.")

    partitions = pd.read_csv(chemin_partitions)
    coords, identifiants, meta_acm = charger_coordonnees_et_identifiants(racine)
    if len(partitions) != taille:
        raise ValueError("Le nombre de lignes du fichier de partition est incorrect.")
    colonne_cible = f"groupe_k{k}"
    if "position_echantillon" not in partitions or colonne_cible not in partitions:
        raise ValueError("Positions ACM ou étiquettes CAH absentes.")
    positions = partitions["position_echantillon"].to_numpy()
    if partitions["position_echantillon"].isna().any() or not pd.api.types.is_integer_dtype(partitions["position_echantillon"]):
        raise ValueError("Les positions doivent être des entiers non manquants.")
    if len(np.unique(positions)) != taille or np.any(positions < 0) or np.any(positions >= len(coords)):
        raise ValueError("Positions ACM non uniques ou hors limites.")
    positions_attendues = selectionner_sous_echantillon(len(coords), taille, graine)
    if not np.array_equal(positions, positions_attendues):
        raise ValueError("Les positions ne correspondent pas au tirage CAH de référence (graine 42).")
    for identifiant in ["id_anonyme", "indice_ligne_source"]:
        if identifiant in identifiants.columns:
            if identifiant not in partitions.columns:
                raise ValueError(f"Identifiant absent des partitions : {identifiant}.")
            attendu = identifiants.iloc[positions][identifiant].reset_index(drop=True)
            observe = partitions[identifiant].reset_index(drop=True)
            if not observe.equals(attendu):
                raise ValueError(f"Correspondance ACM / CAH incorrecte pour {identifiant}.")
    etiquettes = partitions[colonne_cible]
    if etiquettes.isna().any() or set(etiquettes.unique()) != set(range(1, k+1)):
        raise ValueError("Les étiquettes CAH doivent contenir exactement les groupes 1 à 5.")
    if not pd.api.types.is_integer_dtype(etiquettes):
        raise ValueError("Les étiquettes de groupe doivent être entières.")
    colonnes_axes = list(coords.columns[:axes])
    X = coords.iloc[positions, :axes].reset_index(drop=True).copy()
    if X.shape != (taille, axes) or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Coordonnées ACM manquantes ou non finies.")
    y = etiquettes.astype(int).reset_index(drop=True)
    return X, y, partitions.reset_index(drop=True), colonnes_axes, parametres, meta_acm


def decouper_apprentissage(X: pd.DataFrame, y: pd.Series,
                           proportion_test: float = 0.20, graine: int = 42):
    """Sépare les individus, avec stratification sur les cinq groupes."""
    indices = np.arange(len(y))
    return train_test_split(indices, test_size=proportion_test, random_state=graine, stratify=y)


def construire_modeles(graine: int = 42) -> dict:
    """Construit deux modèles de complexité différente, sans fuite de données."""
    return {
        "regression_logistique_multinomiale": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight=None,
                random_state=graine,
            ),
        ),
        # "foret_aleatoire": RandomForestClassifier(
        #     n_estimators=250,
        #     min_samples_leaf=5,
        #     random_state=graine,
        #     n_jobs=-1,
        #     class_weight=None,
        # ),
    }


def evaluer_modele(nom: str, modele, X_test: pd.DataFrame, y_test: pd.Series):
    """Retourne métriques globales et par groupe sur le jeu de test réservé."""
    previsions = modele.predict(X_test)
    rapport = classification_report(y_test, previsions, labels=[1,2,3,4,5],
                                    output_dict=True, zero_division=0)
    bilan = {"modele": nom, "accuracy": accuracy_score(y_test, previsions),
             "f1_macro": f1_score(y_test, previsions, average="macro", zero_division=0),
             "f1_pondere": f1_score(y_test, previsions, average="weighted", zero_division=0)}
    matrice = pd.DataFrame(confusion_matrix(y_test, previsions, labels=[1,2,3,4,5]),
                           index=[f"Réel G{i}" for i in range(1,6)],
                           columns=[f"Prédit G{i}" for i in range(1,6)])
    par_groupe = pd.DataFrame([
        {"groupe": i, "precision": rapport[str(i)]["precision"],
         "rappel": rapport[str(i)]["recall"], "f1": rapport[str(i)]["f1-score"],
         "effectif_test": int(rapport[str(i)]["support"])} for i in range(1,6)
    ])
    return bilan, matrice, par_groupe, previsions
