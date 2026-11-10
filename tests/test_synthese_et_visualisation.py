"""Tests des nouveaux outils de clôture de l'étape 1."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualisation_factorielle import charger_coordonnees_partitions, afficher_partitions_factorielles
from synthese_metier import fiches_groupes, resumer_scissions, bilan_comparatif


def test_appariement_coordonnees_par_position(tmp_path):
    sortie = tmp_path / "resultats"
    sortie.mkdir()
    pd.DataFrame({"Axe 1": [1., 2., 3.], "Axe 2": [5., 6., 7.],
                  "Axe 3": [9., 10., 11.], "Axe 4": [12., 13., 14.] }).to_csv(
        sortie / "coordonnees_factorielles.csv", index=False)
    pd.DataFrame({"position_echantillon": [0, 1, 2], "indice_ligne_source": [4, 8, 9]}).to_csv(
        sortie / "correspondance_clients_acm.csv", index=False)
    partitions = pd.DataFrame({"position_echantillon": [2, 0], "indice_ligne_source": [9, 4],
                               "groupe_k4": [1, 2]})
    retour = charger_coordonnees_partitions(tmp_path, partitions)
    assert retour["Axe 1"].tolist() == [3., 1.]
    partitions.loc[1, "indice_ligne_source"] = 99
    with pytest.raises(ValueError, match="Incohérence"):
        charger_coordonnees_partitions(tmp_path, partitions)


def test_scission_et_bilan():
    donnees = pd.DataFrame({"groupe_k4": [1, 1, 2, 2], "groupe_k5": [1, 3, 2, 2]})
    transitions = donnees.groupby(["groupe_k4", "groupe_k5"]).size().rename("Effectif").reset_index()
    transitions["Part du groupe source (%)"] = [50., 50., 100.]
    resultat = resumer_scissions(transitions, 4, 5)
    assert resultat["Effectif parent"].tolist() == [2, 2]
    assert set(resultat["Groupe enfant"]) == {1, 3}
    tailles = pd.DataFrame([{"k": 4, "Groupe": 1, "Effectif": 2, "Part (%)": 50.},
                            {"k": 4, "Groupe": 2, "Effectif": 2, "Part (%)": 50.},
                            {"k": 5, "Groupe": 1, "Effectif": 1, "Part (%)": 25.},
                            {"k": 5, "Groupe": 2, "Effectif": 2, "Part (%)": 50.},
                            {"k": 5, "Groupe": 3, "Effectif": 1, "Part (%)": 25.}])
    bilan = bilan_comparatif(tailles, resultat, {4:.30, 5:.32})
    assert set(bilan["Décision"]) == {"Non arrêtée"}
    assert bilan.loc[bilan.k == 5, "Silhouette exploratoire"].iloc[0] == .32


def test_fiches_contiennent_frequences():
    profils = pd.DataFrame([
        {"Groupe": 1, "Variable": "type", "Modalité": "A", "Écart (points)": 30.,
         "Effectif modalité groupe": 30, "Part groupe (%)": 70., "Part population (%)": 40.},
        {"Groupe": 1, "Variable": "type", "Modalité": "B", "Écart (points)": -30.,
         "Effectif modalité groupe": 12, "Part groupe (%)": 30., "Part population (%)": 60.},
    ])
    tailles = pd.DataFrame([{"k": 4, "Groupe": 1, "Effectif": 42, "Part (%)": 100.}])
    fiches = fiches_groupes(profils, tailles, 4)
    assert len(fiches) == 2
    assert set(fiches["Type d'écart"]) == {"Surreprésentée", "Sous-représentée"}


def test_graphique_sauvegarde(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    donnees = pd.DataFrame(np.random.default_rng(5).normal(size=(30, 4)))
    groupes = pd.DataFrame({"groupe_k4": np.repeat([1,2,3], 10),
                            "groupe_k5": np.repeat([1,2,3], 10),
                            "groupe_k6": np.repeat([1,2,3], 10)})
    chemin = tmp_path / "figure.png"
    afficher_partitions_factorielles(donnees, groupes, [4,5,6], axes=(3,4),
                                   maximum_points=12, chemin_sortie=chemin)
    assert chemin.exists()
