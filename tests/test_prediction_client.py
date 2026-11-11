"""Tests du moteur de prédiction individuelle, sans artefacts réels."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.prediction_client import PredicteurClient


class FausseACM:
    def row_coordinates(self, disjonctif):
        valeurs = disjonctif.to_numpy(dtype=float)
        return pd.DataFrame(np.column_stack([
            valeurs[:, 0], valeurs[:, 1], valeurs[:, 2], valeurs[:, 3]
        ]))


class FauxClassifieur:
    classes_ = np.arange(1, 6)

    def predict_proba(self, donnees):
        return np.tile([0.04, 0.07, 0.78, 0.06, 0.05], (len(donnees), 1))

    def predict(self, donnees):
        return np.full(len(donnees), 3)


@pytest.fixture
def racine(tmp_path):
    modele = tmp_path / "modeles"
    config = tmp_path / "config"
    modele.mkdir()
    config.mkdir()
    variables = [f"variable_{i}_rec" for i in range(10)]
    modalites = [f"{v}__categorie_{j}" for v in variables for j in range(4)]
    modalites += ["variable_0_rec__categorie_4", "variable_1_rec__categorie_4"]
    with (modele / "acm_final_metadonnees.json").open("w", encoding="utf-8") as f:
        json.dump({"iteration": 3, "variables_actives": variables, "modalites_actives": modalites,
                   "regroupements_acm_finaux": {"variable_0_rec": {"categorie_4": ["ancien_a", "ancien_b"]}}}, f)
    with (modele / "classifieur_clusters_k5_metadonnees.json").open("w", encoding="utf-8") as f:
        json.dump({"modele_selectionne": "regression_logistique_multinomiale",
                   "nombre_axes": 4, "nombre_groupes": 5,
                   "colonnes_axes": [f"Axe {i}" for i in range(1, 5)]}, f)
    regroupements = {v: {"source": f"source_{i}", "correspondances": {
        "brut_0": "categorie_0", "brut_1": "categorie_1",
        "brut_2": "categorie_2", "brut_3": "categorie_3"
    }} for i, v in enumerate(variables)}
    regroupements["variable_0_rec"]["correspondances"]["brut_fusion"] = "ancien_a"
    with (config / "regroupements.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"regroupements": regroupements}, f)
    joblib.dump(FausseACM(), modele / "acm_final.joblib")
    joblib.dump(FauxClassifieur(), modele / "classifieur_clusters_k5.joblib")
    return tmp_path


@pytest.fixture
def client():
    return {**{f"source_{i}": "brut_1" for i in range(10)}, "methode_paiement": "Stripe"}


def test_prediction_unitaire(racine, client):
    predicteur = PredicteurClient(racine)
    resultat = predicteur.predire(client)
    assert resultat.groupe_k5 == 3
    assert resultat.nom_segment == "Clients sans activité observée"
    assert resultat.probabilite_max == pytest.approx(0.78)
    assert sum(resultat.probabilites.values()) == pytest.approx(1)
    assert len(predicteur.champs_requis) == 11


def test_fusion_sur_un_seul_client(racine, client):
    client["source_0"] = "brut_fusion"
    preparation = PredicteurClient(racine).preparer_client(client)
    assert preparation.loc[0, "variable_0_rec"] == "categorie_4"


def test_modalite_inconnue_refusee(racine, client):
    client["source_0"] = "inconnue"
    with pytest.raises(ValueError, match="Modalité inconnue"):
        PredicteurClient(racine).predire(client)


def test_paiement_exclu_refuse(racine, client):
    client["methode_paiement"] = "Others"
    with pytest.raises(ValueError, match="non éligible"):
        PredicteurClient(racine).predire(client)


def test_champ_manquant_refuse(racine, client):
    del client["source_2"]
    with pytest.raises(ValueError, match="Champs manquants"):
        PredicteurClient(racine).predire(client)


def test_artefact_absent_refuse(tmp_path):
    with pytest.raises(FileNotFoundError, match="Artefacts manquants"):
        PredicteurClient(tmp_path)
