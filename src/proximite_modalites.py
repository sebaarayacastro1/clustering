"""Diagnostic de proximité factorielle et de profils conditionnels."""
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from IPython.display import display
import numpy as np
import pandas as pd

def comparer_modalites(echantillon_acm, tableau_modalites, n_axes_proximite=5):
    """Classe les paires sans appliquer de regroupement automatique."""
    # ===== DIAGNOSTIC DE PROXIMITÉ SANS SEUIL ABSOLU ARBITRAIRE =====
    from itertools import combinations
    from scipy.spatial.distance import jensenshannon
    import numpy as np
    import pandas as pd

    # Objets issus de l'itération 2 :
    # echantillon_acm, tableau_modalites et n_axes_proximite.

    # Distribution des autres variables conditionnellement à chaque modalité.
    # Les valeurs manquantes sont traitées comme une catégorie explicite.
    donnees_profils = echantillon_acm.astype('object').fillna('<<MANQUANT>>').astype(str)
    profils_conditionnels = {}
    for variable in donnees_profils.columns:
        profils_conditionnels[variable] = {}
        for autre_variable in donnees_profils.columns:
            if autre_variable != variable:
                profils_conditionnels[variable][autre_variable] = pd.crosstab(
                    donnees_profils[variable],
                    donnees_profils[autre_variable],
                    normalize='index'
                )

    resultats_proximite = []
    for variable, groupe in tableau_modalites.groupby('Variable', sort=False):
        # Même espace factoriel à cinq axes que dans l'analyse précédente.
        colonnes_coordonnees = [f'Coord Axe {i+1}' for i in range(n_axes_proximite)]
        modalites_variable = groupe.reset_index(drop=True)
        for i, j in combinations(range(len(modalites_variable)), 2):
            modalite_a, modalite_b = modalites_variable.iloc[i], modalites_variable.iloc[j]
            libelle_a = str(modalite_a['Modalité'])
            libelle_b = str(modalite_b['Modalité'])
            distance_factorielle = np.linalg.norm(
                modalite_a[colonnes_coordonnees].to_numpy(dtype=float)
                - modalite_b[colonnes_coordonnees].to_numpy(dtype=float)
            )
            distances_js = []
            for autre_variable, tableau_profil in profils_conditionnels[variable].items():
                if libelle_a in tableau_profil.index and libelle_b in tableau_profil.index:
                    profil_a = tableau_profil.loc[libelle_a].to_numpy(dtype=float)
                    profil_b = tableau_profil.loc[libelle_b].to_numpy(dtype=float)
                    distance_js = jensenshannon(profil_a, profil_b, base=2)
                    if np.isfinite(distance_js):
                        distances_js.append(distance_js)
            resultats_proximite.append({
                'Variable': variable,
                'Modalité A': libelle_a,
                'Modalité B': libelle_b,
                'Effectif A': int(modalite_a['Effectif']),
                'Effectif B': int(modalite_b['Effectif']),
                'Distance factorielle': float(distance_factorielle),
                'JS moyenne': float(np.mean(distances_js)) if distances_js else np.nan,
                'JS maximale': float(np.max(distances_js)) if distances_js else np.nan,
            })

    comparaison_modalites = pd.DataFrame(resultats_proximite)
    if comparaison_modalites.empty:
        print('Aucune paire de modalités à comparer.')
    else:
        # Classement relatif par variable : les valeurs les plus faibles sont les plus proches.
        for colonne_source, colonne_rang in [
            ('Distance factorielle', 'Rang relatif factoriel'),
            ('JS moyenne', 'Rang relatif JS'),
        ]:
            comparaison_modalites[colonne_rang] = (
                comparaison_modalites.groupby('Variable')[colonne_source]
                .rank(method='min', pct=True, ascending=True)
            )

        # Indicateur de tri exploratoire, et non critère automatique de regroupement.
        comparaison_modalites['Priorite exploratoire'] = (
            comparaison_modalites['Rang relatif factoriel']
            + comparaison_modalites['Rang relatif JS']
        ) / 2
        comparaison_modalites = comparaison_modalites.sort_values(
            ['Priorite exploratoire', 'Distance factorielle']
        ).reset_index(drop=True)

        print('Paires les mieux classées selon les deux distances (à examiner) :')
        display(comparaison_modalites.head(25).round(4))

        print('Paire la plus proche dans l’espace factoriel pour chaque variable :')
        indices_minimum = comparaison_modalites.groupby('Variable')['Distance factorielle'].idxmin()
        display(comparaison_modalites.loc[indices_minimum].sort_values('Variable').round(4))
    return comparaison_modalites
