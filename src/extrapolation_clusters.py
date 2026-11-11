"""Projection ACM figée et extrapolation des cinq segments à la base source."""

from __future__ import annotations

import numpy as np
import pandas as pd


def verifier_correspondance_source(donnees_brutes, donnees_nettoyees, correspondance, *, graine=42, fraction=0.20):
    """Vérifie l'échantillonnage original et sa correspondance avec les lignes sources."""
    if not donnees_brutes.index.equals(pd.RangeIndex(len(donnees_brutes))):
        raise ValueError("L'index brut doit correspondre aux positions physiques du CSV.")
    if not donnees_nettoyees.index.is_unique:
        raise ValueError("L'index des données préparées n'est pas unique.")
    positions = donnees_nettoyees.sample(frac=fraction, random_state=graine).index.to_numpy()
    if len(positions) != len(correspondance):
        raise ValueError("La taille de l'échantillon reconstruit diffère de l'ACM.")
    if "position_echantillon" not in correspondance or not np.array_equal(
        correspondance["position_echantillon"].to_numpy(), np.arange(len(positions))
    ):
        raise ValueError("Les positions de l'échantillon ACM ne sont pas cohérentes.")
    if "indice_ligne_source" in correspondance:
        if not np.array_equal(positions, correspondance["indice_ligne_source"].to_numpy()):
            raise ValueError("L'échantillon reconstruit ne correspond pas aux indices sources.")
    elif "id_anonyme" in correspondance:
        if "id_anonyme" not in donnees_brutes:
            raise ValueError("Identifiants métier manquants dans le CSV source.")
        attendus = donnees_brutes.loc[positions, "id_anonyme"].reset_index(drop=True)
        if not attendus.equals(correspondance["id_anonyme"].reset_index(drop=True)):
            raise ValueError("Les identifiants source ne correspondent pas à ceux de l'ACM.")
    else:
        raise ValueError("Aucune clé de correspondance source n'est disponible.")
    return positions


def preparer_tableau_disjonctif(donnees, variables_actives, modalites_actives):
    """Reproduit le tableau disjonctif du modèle sans ajouter de modalités inédites."""
    manquantes = sorted(set(variables_actives) - set(donnees.columns))
    if manquantes:
        raise ValueError(f"Variables actives manquantes : {manquantes}")
    variables = donnees.loc[:, variables_actives]
    if variables.isna().any().any():
        compte = int(variables.isna().any(axis=1).sum())
        raise ValueError(f"{compte} client(s) présentent des modalités actives manquantes.")
    tableau = pd.get_dummies(variables, prefix_sep="__").astype(float)
    inconnues = sorted(set(tableau.columns) - set(modalites_actives))
    if inconnues:
        raise ValueError(f"Modalités inédites par rapport à l'ACM : {inconnues}")
    tableau = tableau.reindex(columns=modalites_actives, fill_value=0.0)
    if not np.allclose(tableau.sum(axis=1).to_numpy(), len(variables_actives)):
        raise ValueError("Le tableau disjonctif ne contient pas une modalité par variable.")
    return tableau


def projeter_individus(modele_acm, donnees, variables_actives, modalites_actives, nombre_axes=4, taille_lot=5000):
    """Calcule uniquement row_coordinates() : aucun ajustement de l'ACM."""
    if taille_lot < 1:
        raise ValueError("La taille des lots doit être positive.")
    if len(donnees) == 0:
        return pd.DataFrame(index=donnees.index)
    blocs = []
    for debut in range(0, len(donnees), taille_lot):
        lot = donnees.iloc[debut:debut + taille_lot]
        tableau = preparer_tableau_disjonctif(lot, variables_actives, modalites_actives)
        coordonnees = modele_acm.row_coordinates(tableau).iloc[:, :nombre_axes]
        coordonnees = coordonnees.copy()
        coordonnees.index = lot.index
        if coordonnees.shape != (len(lot), nombre_axes) or not np.isfinite(coordonnees.to_numpy(dtype=float)).all():
            raise ValueError("Projection ACM invalide.")
        blocs.append(coordonnees)
    return pd.concat(blocs, axis=0)


def predire_par_lots(classifieur, coordonnees, colonnes_axes, *, taille_lot=5000):
    """Prédit les groupes et probabilités sans réentraîner le classifieur."""
    if taille_lot < 1:
        raise ValueError("La taille des lots doit être positive.")
    if len(coordonnees) == 0:
        raise ValueError("Aucune coordonnée à prédire.")
    classes = np.asarray(classifieur.classes_, dtype=int)
    if not np.array_equal(classes, np.arange(1, 6)):
        raise ValueError("Les classes du modèle ne sont pas G1 à G5.")
    predictions = []
    probabilities = []
    for debut in range(0, len(coordonnees), taille_lot):
        X = coordonnees.iloc[debut:debut + taille_lot].copy()
        X.columns = colonnes_axes
        y = np.asarray(classifieur.predict(X), dtype=int)
        proba = np.asarray(classifieur.predict_proba(X), dtype=float)
        if (proba.shape != (len(X), 5) or not np.isfinite(proba).all()
                or not np.allclose(proba.sum(axis=1), 1, atol=1e-8)
                or not np.array_equal(y, classes[proba.argmax(axis=1)])):
            raise ValueError("Prédictions ou probabilités incohérentes.")
        predictions.append(y)
        probabilities.append(proba)
    return np.concatenate(predictions), np.vstack(probabilities)


def verifier_couverture_finale(base_finale, *, nombre_brut, nombre_acm, nombre_hors_acm, nombre_exclus):
    """Garantit une ligne par client brut, sans affectation des cas exclus."""
    if len(base_finale) != nombre_brut:
        raise ValueError("La table finale ne conserve pas toutes les lignes brutes.")
    if not np.array_equal(base_finale["indice_ligne_source"].to_numpy(), np.arange(nombre_brut)):
        raise ValueError("Ordre ou unicité des lignes sources incorrects.")
    origine = base_finale["origine_affectation"]
    if int(origine.isin(["CAH_reference", "regression_logistique_multinomiale"]).sum()) != nombre_acm:
        raise ValueError("Le nombre d'individus de l'échantillon ACM diffère.")
    if int((origine == "projection_ACM_et_regression_logistique").sum()) != nombre_hors_acm:
        raise ValueError("Le nombre de projections hors ACM diffère.")
    if int((origine == "exclu_preparation").sum()) != nombre_exclus:
        raise ValueError("Le nombre d'exclus diffère.")
    eligibles = origine != "exclu_preparation"
    if not base_finale.loc[eligibles, "groupe_k5"].isin(range(1, 6)).all():
        raise ValueError("Un client éligible n'a pas de groupe valide.")
    if base_finale.loc[~eligibles, "groupe_k5"].notna().any():
        raise ValueError("Un client exclu a reçu une étiquette sans justification.")
