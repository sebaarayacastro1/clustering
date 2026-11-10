"""Tests unitaires des outils d'interprétation des partitions."""
import numpy as np
import pandas as pd
import pytest
from src.interpretation import (
    choisir_variables_descriptives, tailles_partitions, profils_modalites,
    transitions_groupes, differences_enfants,
)


def jeu_exemple():
    donnees = pd.DataFrame({
        "activite": ["faible", "faible", "forte", "forte", "faible", "forte"],
        "valeur": [1, 2, 4, 6, 1, 10],
        "id_anonyme": list(range(6)),
    })
    partitions = pd.DataFrame({
        "groupe_k2": [1, 1, 1, 2, 2, 2],
        "groupe_k3": [1, 1, 1, 2, 3, 3],
    })
    return donnees, partitions


def test_selection_variables():
    donnees, _ = jeu_exemple()
    qualitatives, quantitatives = choisir_variables_descriptives(donnees, maximum_modalites=3)
    assert qualitatives == ["activite"]
    assert quantitatives == ["valeur"]


def test_effectifs():
    _, partitions = jeu_exemple()
    tailles = tailles_partitions(partitions, [2, 3])
    assert tailles.loc[tailles["k"] == 2, "Effectif"].sum() == 6
    assert tailles.loc[tailles["k"] == 3, "Effectif"].sum() == 6


def test_profils():
    donnees, partitions = jeu_exemple()
    profils = profils_modalites(donnees, partitions["groupe_k2"].to_numpy(), ["activite"])
    ligne = profils.query("Groupe == 1 and Modalité == 'faible'").iloc[0]
    assert ligne["Part groupe (%)"] == pytest.approx(200 / 3)
    assert ligne["Part population (%)"] == pytest.approx(50.0)


def test_transition():
    _, partitions = jeu_exemple()
    transitions = transitions_groupes(partitions, 2, 3)
    assert transitions["Effectif"].sum() == 6
    assert len(transitions) == 3


def test_differences_enfants():
    donnees, partitions = jeu_exemple()
    differences = differences_enfants(donnees, partitions, 2, 3, ["activite"])
    assert not differences.empty
    assert set(differences["Groupe parent"]) == {2}
