"""Contrôles d'intégration de la prédiction individuelle avec l'étape 3.

Ces tests exigent les artefacts locaux et les exports historiques. Ils ne
réentraînent aucun modèle et ne modifient aucun fichier du projet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction_client import PredicteurClient


RACINE = Path(__file__).resolve().parents[1]
CHEMIN_SOURCE = RACINE / "data" / "brutes" / "td_clustering_karafun.csv"
CHEMIN_REFERENCE = (
    RACINE / "resultats" / "extrapolation" / "base_complete_segmentee.csv"
)
ARTEFACTS = (
    RACINE / "modeles" / "acm_final.joblib",
    RACINE / "modeles" / "acm_final_metadonnees.json",
    RACINE / "modeles" / "classifieur_clusters_k5.joblib",
    RACINE / "modeles" / "classifieur_clusters_k5_metadonnees.json",
    RACINE / "config" / "regroupements.yaml",
)
ORIGINE_HORS_ACM = "projection_ACM_et_regression_logistique"
COLONNES_PROBABILITES = [f"probabilite_G{groupe}" for groupe in range(1, 6)]


@pytest.fixture(scope="module")
def jeu_historique():
    """Charge les données réelles uniquement si tous les fichiers existent."""
    chemins = (CHEMIN_SOURCE, CHEMIN_REFERENCE, *ARTEFACTS)
    absents = [str(p.relative_to(RACINE)) for p in chemins if not p.is_file()]
    if absents:
        pytest.skip("Contrôle réel indisponible ; fichiers absents : " + ", ".join(absents))

    brut = pd.read_csv(CHEMIN_SOURCE)
    reference = pd.read_csv(CHEMIN_REFERENCE)
    if len(brut) != 341_092 or len(reference) != 341_092:
        pytest.fail("La base source ou l'export n'a pas 341 092 lignes.")
    indices = reference["indice_ligne_source"].to_numpy(dtype=int)
    if not np.array_equal(indices, np.arange(len(brut))):
        pytest.fail("La référence ne respecte pas l'ordre du CSV source.")
    if reference["indice_ligne_source"].duplicated().any():
        pytest.fail("Indices sources dupliqués dans la référence.")

    hors_acm = reference.loc[
        reference["origine_affectation"].eq(ORIGINE_HORS_ACM)
    ]
    if len(hors_acm) != 272_873:
        pytest.fail("Le nombre de clients projetés hors ACM est inattendu.")
    if not set(hors_acm["groupe_k5"].astype(int).unique()) == set(range(1, 6)):
        pytest.fail("La référence ne contient pas les cinq groupes hors ACM.")

    # Échantillon équilibré, déterministe et dispersé dans chaque groupe.
    selections = []
    for groupe in range(1, 6):
        candidats = hors_acm.loc[hors_acm["groupe_k5"].eq(groupe)]
        if len(candidats) < 3:
            pytest.fail(f"Pas assez de clients de contrôle dans G{groupe}.")
        emplacements = np.linspace(0, len(candidats) - 1, num=3, dtype=int)
        selections.append(candidats.iloc[emplacements])
    echantillon = pd.concat(selections).sort_values("indice_ligne_source")
    return brut, echantillon, PredicteurClient(RACINE)


def _predictions(jeu_historique):
    brut, echantillon, predicteur = jeu_historique
    resultats = []
    for ligne in echantillon.itertuples(index=False):
        indice = int(ligne.indice_ligne_source)
        resultat = predicteur.predire(brut.iloc[indice].to_dict())
        resultats.append((ligne, resultat))
    return resultats


def test_groupes_et_noms_identiques_a_l_extrapolation(jeu_historique):
    """La prédiction individuelle reproduit 15 affectations de la partie B."""
    for ligne, resultat in _predictions(jeu_historique):
        assert resultat.groupe_k5 == int(ligne.groupe_k5), (
            f"Client source {ligne.indice_ligne_source} : "
            f"G{resultat.groupe_k5} au lieu de G{int(ligne.groupe_k5)}."
        )
        assert resultat.nom_segment == ligne.nom_segment


def test_probabilites_identiques_a_l_extrapolation(jeu_historique):
    """Les cinq scores et le score maximal reproduisent l'export historique."""
    for ligne, resultat in _predictions(jeu_historique):
        attendues = np.array(
            [float(getattr(ligne, colonne)) for colonne in COLONNES_PROBABILITES]
        )
        obtenues = np.array(
            [resultat.probabilites[f"G{groupe}"] for groupe in range(1, 6)]
        )
        np.testing.assert_allclose(
            obtenues, attendues, rtol=1e-7, atol=1e-9,
            err_msg=f"Probabilités différentes pour la ligne {ligne.indice_ligne_source}.",
        )
        assert resultat.probabilite_max == pytest.approx(
            float(ligne.probabilite_max), abs=1e-9
        )
        assert obtenues.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.isfinite(obtenues).all()


def test_prediction_independante_de_l_identifiant_et_de_l_ordre(jeu_historique):
    """Les colonnes inutiles et l'ordre des champs ne modifient pas le résultat."""
    brut, echantillon, predicteur = jeu_historique
    indice = int(echantillon.iloc[0]["indice_ligne_source"])
    client = brut.iloc[indice].to_dict()
    resultat_initial = predicteur.predire(client)

    client_reordonne = dict(reversed(list(client.items())))
    client_reordonne["indice_ligne_source"] = 999_999
    client_reordonne["groupe_k5"] = 1  # Ne doit jamais servir d'entrée prédictive.
    resultat_reordonne = predicteur.predire(client_reordonne)
    assert resultat_reordonne.groupe_k5 == resultat_initial.groupe_k5
    assert resultat_reordonne.nom_segment == resultat_initial.nom_segment
    for groupe in range(1, 6):
        assert resultat_reordonne.probabilites[f"G{groupe}"] == pytest.approx(
            resultat_initial.probabilites[f"G{groupe}"], abs=1e-12
        )
