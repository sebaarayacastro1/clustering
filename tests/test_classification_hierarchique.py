"""Tests unitaires sur des jeux de données artificiels, sans le CSV réel."""

import numpy as np
import pandas as pd
import pytest
from src.classification_hierarchique import (
    choisir_axes, selectionner_sous_echantillon,
    estimer_memoire_distances, effectuer_cah, obtenir_partition,
    tailles_groupes, hauteurs_fusions,
)
from src.evaluation_clusters import calculer_silhouette_echantillon, comparer_partitions
from src.interpretation import profils_modalites


def jeu_exemple():
    return pd.DataFrame({"Axe 1": [-5, -4.9, -5.2, 4.9, 5, 5.1],
                         "Axe 2": [0, 0.1, -0.1, 0, 0.1, -0.1],
                         "Axe 3": [1, 1, 1, 1, 1, 1]})


def test_choix_axes_et_echantillonnage():
    donnees = jeu_exemple()
    assert choisir_axes(donnees, 2).shape == (6, 2)
    assert np.array_equal(selectionner_sous_echantillon(6, 6), np.arange(6))
    assert np.array_equal(selectionner_sous_echantillon(6, 3),
                          selectionner_sous_echantillon(6, 3))
    with pytest.raises(ValueError):
        choisir_axes(donnees, 4)


def test_memoire_condensee():
    estimation = estimer_memoire_distances(100)
    assert estimation["paires"] == 4950
    assert estimation["memoire_minimale_gio"] > 0


def test_ward_et_partitions():
    donnees = choisir_axes(jeu_exemple(), 2)
    hierarchie = effectuer_cah(donnees, budget_memoire_gio=0.001)
    assert hierarchie.shape == (5, 4)
    etiquettes = obtenir_partition(hierarchie, 2)
    assert len(set(etiquettes)) == 2
    assert len(set(etiquettes[:3])) == 1
    assert len(set(etiquettes[3:])) == 1
    assert tailles_groupes(etiquettes, "k=2")["Effectif"].sum() == 6
    assert len(hauteurs_fusions(hierarchie, nombre_dernieres=3)) == 3
    assert calculer_silhouette_echantillon(donnees, etiquettes, taille_maximale=6) > 0


def test_refus_memoire():
    with pytest.raises(MemoryError):
        effectuer_cah(jeu_exemple(), budget_memoire_gio=0.000000001)


def test_profil_et_comparaison():
    groupes = np.array([1, 1, 1, 2, 2, 2])
    profils = profils_modalites(pd.DataFrame({"Catégorie": ["A"] * 3 + ["B"] * 3}),
                                groupes, ["Catégorie"])
    assert len(profils) == 4
    assert comparer_partitions(groupes, groupes, "A", "B")["ARI"].iloc[0] == 1.0
