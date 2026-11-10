"""Tests de détection des variables descriptives."""

import pandas as pd
from src.interpretation import choisir_variables_descriptives


def test_colonnes_textuelles_et_identifiant():
    donnees = pd.DataFrame({
        "id_anonyme": ["a", "b", "c"],
        "type_client": pd.Series(["A", "B", "A"], dtype="string"),
        "anciennete_client_tr": ["0-1", "1-2", "2-3"],
        "nb_abo_periode": [1, 2, 3],
    })
    qualitatives, quantitatives = choisir_variables_descriptives(donnees)
    assert "id_anonyme" not in qualitatives + quantitatives
    assert {"type_client", "anciennete_client_tr", "nb_abo_periode"}.issubset(qualitatives)


def test_cardinalite_et_variables_quantitatives():
    donnees = pd.DataFrame({
        "categorie": [f"M{i}" for i in range(20)],
        "montant": [float(i) for i in range(20)],
    })
    qualitatives, quantitatives = choisir_variables_descriptives(donnees, maximum_modalites=25)
    assert "categorie" in qualitatives
    assert "montant" in qualitatives  # Variable numérique discrète à faible cardinalité
    qualitatives, quantitatives = choisir_variables_descriptives(donnees, maximum_modalites=15)
    assert "categorie" not in qualitatives
    assert "montant" in quantitatives
