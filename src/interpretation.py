"""Statistiques descriptives des groupes, sans attribuer de sens métier automatique."""

from __future__ import annotations
import numpy as np
import pandas as pd


def profils_modalites(donnees: pd.DataFrame, etiquettes: np.ndarray,
                       variables: list[str]) -> pd.DataFrame:
    """Taux de chaque modalité par groupe et dans la population étudiée."""
    if len(donnees) != len(etiquettes):
        raise ValueError("L'ordre et le nombre d'individus doivent être identiques.")
    if not variables:
        raise ValueError("Sélectionner au moins une variable descriptive.")
    manquantes = set(variables) - set(donnees.columns)
    if manquantes:
        raise KeyError(f"Variables absentes : {sorted(manquantes)}")
    table = donnees[variables].reset_index(drop=True).copy()
    table["Groupe"] = np.asarray(etiquettes)
    lignes = []
    for variable in variables:
        for modalite, part_population in table[variable].value_counts(normalize=True, dropna=False).items():
            masque = table[variable].eq(modalite)
            for groupe, groupe_table in table.groupby("Groupe", sort=True):
                taux = 100 * groupe_table[variable].eq(modalite).mean()
                lignes.append({"Variable": variable, "Modalité": str(modalite),
                               "Groupe": int(groupe), "Part groupe (%)": taux,
                               "Part population (%)": 100 * part_population,
                               "Écart (points)": taux - 100 * part_population,
                               "Effectif groupe": len(groupe_table)})
    return pd.DataFrame(lignes)
