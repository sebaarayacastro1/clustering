"""Fiches descriptives et suivi des scissions pour choisir une partition CAH."""
from __future__ import annotations
import numpy as np
import pandas as pd


def fiches_groupes(profils: pd.DataFrame, tailles: pd.DataFrame, k: int,
                   nombre_modalites: int = 6, effectif_minimal: int = 10) -> pd.DataFrame:
    """Résume les écarts positifs et négatifs sans inventer de nom métier."""
    sous_tailles = tailles.loc[tailles["k"] == k].set_index("Groupe")
    lignes = []
    for groupe in sorted(sous_tailles.index):
        donnees_groupe = profils.loc[profils["Groupe"] == groupe].copy()
        positives = (donnees_groupe.loc[
            (donnees_groupe["Écart (points)"] > 0) &
            (donnees_groupe["Effectif modalité groupe"] >= effectif_minimal)]
            .sort_values("Écart (points)", ascending=False).head(nombre_modalites))
        negatives = (donnees_groupe.loc[donnees_groupe["Écart (points)"] < 0]
                     .sort_values("Écart (points)").head(nombre_modalites))
        for statut, tableau in [("Surreprésentée", positives), ("Sous-représentée", negatives)]:
            for _, item in tableau.iterrows():
                lignes.append({"k": k, "Groupe": int(groupe),
                               "Effectif groupe": int(sous_tailles.loc[groupe, "Effectif"]),
                               "Part groupe total (%)": float(sous_tailles.loc[groupe, "Part (%)"]),
                               "Type d'écart": statut, "Variable": item["Variable"],
                               "Modalité": item["Modalité"],
                               "Fréquence dans le groupe (%)": float(item["Part groupe (%)"]),
                               "Fréquence totale (%)": float(item["Part population (%)"]),
                               "Écart (points)": float(item["Écart (points)"])})
    return pd.DataFrame(lignes)


def resumer_scissions(transitions: pd.DataFrame, k_source: int, k_cible: int) -> pd.DataFrame:
    """Décrit automatiquement la scission unique entre partitions successives."""
    if k_cible != k_source + 1:
        raise ValueError("Les valeurs de k doivent être consécutives.")
    colonne_source, colonne_cible = f"groupe_k{k_source}", f"groupe_k{k_cible}"
    effectifs = transitions.groupby(colonne_source)[colonne_cible].nunique()
    scindes = effectifs[effectifs == 2]
    if len(scindes) != 1 or (effectifs > 2).any():
        raise ValueError("La partition n'exprime pas une scission hiérarchique unique.")
    parent = int(scindes.index[0])
    lignes = transitions[transitions[colonne_source] == parent].sort_values("Effectif", ascending=False)
    return pd.DataFrame([{"Passage": f"{k_source} → {k_cible}", "Groupe parent": parent,
                          "Effectif parent": int(lignes["Effectif"].sum()),
                          "Groupe enfant": int(row[colonne_cible]), "Effectif enfant": int(row["Effectif"]),
                          "Part du parent (%)": float(row["Part du groupe source (%)"])}
                         for _, row in lignes.iterrows()])


def bilan_comparatif(tailles: pd.DataFrame, scissions: pd.DataFrame,
                      silhouettes: dict[int, float] | None = None) -> pd.DataFrame:
    """Consolide des faits observés et laisse la décision métier explicite."""
    silhouettes = silhouettes or {}
    lignes = []
    for k, sous_table in tailles.groupby("k", sort=True):
        precedentes = scissions.loc[scissions["Passage"] == f"{k-1} → {k}"] if not scissions.empty else pd.DataFrame()
        description = "Partition de référence" if precedentes.empty else (
            f"G{int(precedentes.iloc[0]['Groupe parent'])} → " + " + ".join(
                f"G{int(r['Groupe enfant'])} ({int(r['Effectif enfant'])})"
                for _, r in precedentes.iterrows()))
        lignes.append({"k": int(k), "Nombre de clients": int(sous_table["Effectif"].sum()),
                       "Plus petit groupe": int(sous_table["Effectif"].min()),
                       "Part minimale (%)": float(sous_table["Part (%)"].min()),
                       "Silhouette exploratoire": silhouettes.get(int(k), np.nan),
                       "Scission supplémentaire": description,
                       "Interprétabilité métier": "À argumenter à partir des fiches",
                       "Stabilité": "À vérifier sur un autre échantillon",
                       "Décision": "Non arrêtée"})
    return pd.DataFrame(lignes)
