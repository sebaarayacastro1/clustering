"""Contrôles unitaires de l'extrapolation des segments CAH."""

import numpy as np
import pandas as pd
import pytest

from src.extrapolation_clusters import (
    predire_par_lots,
    preparer_tableau_disjonctif,
    projeter_individus,
    verifier_correspondance_source,
    verifier_couverture_finale,
)


MODALITES = ["a__x", "a__y", "b__p", "b__q"]


def test_tableau_disjonctif_ordre_et_codage():
    donnees = pd.DataFrame({"a": ["y", "x"], "b": ["p", "q"]})
    tableau = preparer_tableau_disjonctif(donnees, ["a", "b"], MODALITES)
    assert list(tableau.columns) == MODALITES
    assert tableau.to_numpy().tolist() == [[0., 1., 1., 0.], [1., 0., 0., 1.]]


def test_modalite_inconnue_refusee():
    donnees = pd.DataFrame({"a": ["inconnue"], "b": ["p"]})
    with pytest.raises(ValueError, match="Modalités inédites"):
        preparer_tableau_disjonctif(donnees, ["a", "b"], MODALITES)


def test_modalite_absente_refusee():
    donnees = pd.DataFrame({"a": [None], "b": ["p"]})
    with pytest.raises(ValueError, match="manquantes"):
        preparer_tableau_disjonctif(donnees, ["a", "b"], MODALITES)


def test_correspondance_source_reconstituee():
    brut = pd.DataFrame({"id_anonyme": list("abcdefghij")})
    prepares = brut.drop(index=[2])
    positions = prepares.sample(frac=0.2, random_state=42).index.to_numpy()
    correspondance = pd.DataFrame({
        "position_echantillon": range(len(positions)),
        "indice_ligne_source": positions,
    })
    assert np.array_equal(verifier_correspondance_source(brut, prepares, correspondance), positions)
    correspondance.loc[0, "indice_ligne_source"] = 999
    with pytest.raises(ValueError, match="indices sources"):
        verifier_correspondance_source(brut, prepares, correspondance)


class FauxACM:
    def row_coordinates(self, donnees):
        return pd.DataFrame(donnees.iloc[:, :2].to_numpy(), index=donnees.index)


class FauxClassifieur:
    classes_ = np.arange(1, 6)

    def predict_proba(self, donnees):
        probabilites = np.zeros((len(donnees), 5))
        probabilites[:, 0] = 0.8
        probabilites[:, 1:] = 0.05
        return probabilites

    def predict(self, donnees):
        return np.ones(len(donnees), dtype=int)


def test_projection_et_prediction_par_lots():
    donnees = pd.DataFrame({"a": ["x", "y", "x"], "b": ["p", "p", "q"]})
    coordonnees = projeter_individus(FauxACM(), donnees, ["a", "b"], MODALITES,
                                    nombre_axes=2, taille_lot=2)
    etiquettes, proba = predire_par_lots(FauxClassifieur(), coordonnees, ["Axe 1", "Axe 2"],
                                         taille_lot=2)
    assert len(etiquettes) == 3
    assert np.array_equal(etiquettes, [1, 1, 1])
    assert np.allclose(proba.sum(axis=1), 1)


def test_couverture_et_exclusion():
    base = pd.DataFrame({
        "indice_ligne_source": [0, 1, 2, 3],
        "origine_affectation": ["CAH_reference", "regression_logistique_multinomiale",
                                "projection_ACM_et_regression_logistique", "exclu_preparation"],
        "groupe_k5": pd.Series([1, 2, 3, pd.NA], dtype="Int64"),
    })
    verifier_couverture_finale(base, nombre_brut=4, nombre_acm=2,
                               nombre_hors_acm=1, nombre_exclus=1)
    base.loc[3, "groupe_k5"] = 4
    with pytest.raises(ValueError, match="exclu"):
        verifier_couverture_finale(base, nombre_brut=4, nombre_acm=2,
                                   nombre_hors_acm=1, nombre_exclus=1)
