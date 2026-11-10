"""Graphiques descriptifs pour l'interprétation des groupes de clients."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def afficher_tailles_groupes(tailles: pd.DataFrame) -> None:
    """Affiche les parts de groupes par partition en barres empilées."""
    fig, axe = plt.subplots(figsize=(11, 4))
    partitions = sorted(tailles["k"].unique())
    for position, k in enumerate(partitions):
        sous_table = tailles.loc[tailles["k"] == k].sort_values("Groupe")
        debut = 0.0
        for _, ligne in sous_table.iterrows():
            part = float(ligne["Part (%)"])
            axe.barh(position, part, left=debut, edgecolor="white", linewidth=0.8)
            if part >= 8:
                axe.text(debut + part / 2, position, f"{part:.1f} %", ha="center", va="center", fontsize=9)
            debut += part
    axe.set_yticks(range(len(partitions)), [f"k = {k}" for k in partitions])
    axe.set_xlim(0, 100)
    axe.set_xlabel("Part des individus (%)")
    axe.set_title("Répartition des individus selon le nombre de groupes")
    axe.invert_yaxis()
    fig.tight_layout()
    plt.show()


def afficher_carte_ecarts(profils: pd.DataFrame, k: int, nombre_modalites: int = 20) -> None:
    """Affiche les écarts en points aux fréquences de l'ensemble de l'échantillon."""
    if profils.empty:
        print(f"Aucun profil disponible pour k={k}.")
        return
    profils = profils.copy()
    profils["Libellé"] = profils["Variable"].astype(str) + " : " + profils["Modalité"].astype(str)
    priorites = (profils.groupby("Libellé")["Écart (points)"].apply(lambda x: x.abs().max())
                 .sort_values(ascending=False).head(nombre_modalites).index)
    matrice = (profils[profils["Libellé"].isin(priorites)]
               .pivot(index="Libellé", columns="Groupe", values="Écart (points)")
               .reindex(priorites))
    limite = max(1.0, float(np.nanmax(np.abs(matrice.to_numpy()))))
    fig, axe = plt.subplots(figsize=(max(7, len(matrice.columns) * 1.35), max(5, len(matrice) * 0.34)))
    image = axe.imshow(matrice, cmap="RdBu_r", aspect="auto", vmin=-limite, vmax=limite)
    axe.set_xticks(range(len(matrice.columns)), [f"G{x}" for x in matrice.columns], fontsize=10)
    axe.set_xlabel("Groupes de la CAH")
    axe.set_yticks(range(len(matrice.index)), matrice.index, fontsize=9)
    axe.set_title(f"Modalités sur- et sous-représentées — k = {k}")
    fig.colorbar(image, ax=axe, label="Écart à la population (points)")
    fig.tight_layout()
    plt.show()


def afficher_transitions(tableau: pd.DataFrame, k_source: int, k_cible: int) -> None:
    """Présente les effectifs de passage entre deux découpages de la hiérarchie."""
    source, cible = f"groupe_k{k_source}", f"groupe_k{k_cible}"
    matrice = tableau.pivot(index=source, columns=cible, values="Effectif").fillna(0)
    fig, axe = plt.subplots(figsize=(max(6, len(matrice.columns) * 1.1), max(4, len(matrice) * 0.8)))
    axe.imshow(matrice, cmap="Blues", aspect="auto")
    axe.set_xticks(range(len(matrice.columns)), [str(v) for v in matrice.columns])
    axe.set_yticks(range(len(matrice.index)), [str(v) for v in matrice.index])
    for i in range(len(matrice.index)):
        for j in range(len(matrice.columns)):
            axe.text(j, i, str(int(matrice.iat[i, j])), ha="center", va="center", fontsize=9)
    axe.set(xlabel=f"Groupes (k={k_cible})", ylabel=f"Groupes (k={k_source})",
            title=f"Transition des groupes : k={k_source} vers k={k_cible}")
    fig.tight_layout()
    plt.show()
