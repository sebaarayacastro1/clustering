"""Analyse des correspondances multiples, itérations historiques 1 et 2."""
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prince
from IPython.display import display
from .visualisation import surligner_modalites_par_axe, surligner_resume, surligner_proximite

# ===== FONCTIONS DE CALCUL ET D'INERTIE =====
def variable_de(modalite):
    """Retourne le nom de la variable à partir du nom de la colonne (variable__modalite)."""
    return modalite.split('__')[0]

def inertie_brute(valeurs_propres, inertie_totale):
    """Inertie brute (en %) de chaque axe : valeur propre / inertie totale."""
    return np.array(valeurs_propres) / inertie_totale * 100

def inertie_benzecri(valeurs_propres, nb_var):
    """Inertie corrigée de Benzécri (en %) pour chaque axe."""
    seuil = 1 / nb_var
    corrigees = np.array([
        ((nb_var / (nb_var - 1)) ** 2) * ((lam - seuil) ** 2) if lam > seuil else 0
        for lam in valeurs_propres
    ])
    return corrigees / corrigees.sum() * 100


def executer_iteration_1(donnees_nettoyees, variables_a_retirer, *, graine=42, numero_iteration=1, n_axes_tableau=5, n_axes_proximite=5, facteur_seuil=1.0, seuil_proximite=0.20, variables_exclues=None):
    """Reproduit les tableaux, diagnostics et graphiques de l’itération historique."""
    if variables_exclues is None:
        variables_exclues = ["id_anonyme", "statut_abonnement"]
    # ITERATION 1

    # ===== 1. ÉCHANTILLON DE 20 % (mêmes individus à chaque itération) =====
    variables_acm = [
        col for col in donnees_nettoyees.select_dtypes(include=['object', 'category']).columns
        if col not in variables_exclues + variables_a_retirer
    ]
    indices_echantillon = donnees_nettoyees.sample(frac=0.20, random_state=graine).index
    echantillon_acm = donnees_nettoyees.loc[indices_echantillon, variables_acm]
    print(f"Itération {numero_iteration} : {len(echantillon_acm)} individus, {len(variables_acm)} variables")

    # ===== 2. TABLEAU DISJONCTIF COMPLET =====
    tableau_disjonctif = pd.get_dummies(
        echantillon_acm, prefix_sep='__'
    ).astype(float)

    nb_modalites = tableau_disjonctif.shape[1]
    nb_variables = len(variables_acm)

    # Nombre maximal d'axes factoriels non triviaux
    nombre_axes_max = nb_modalites - nb_variables
    n_composantes = min(nombre_axes_max, len(echantillon_acm) - 1)

    print(f"Nombre de modalités actives (K) : {nb_modalites}")
    print(f"Nombre d'axes factoriels calculés : {n_composantes}")

    # ===== 3. AJUSTEMENT DE L'ACM (AFC sur le tableau disjonctif) =====
    modele = prince.CA(
        n_components=n_composantes,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=graine
    )
    modele = modele.fit(tableau_disjonctif)

    # ===== 4. TABLEAU D'INERTIE =====
    valeurs_propres = np.array(modele.eigenvalues_)
    inertie_totale = (nb_modalites - nb_variables) / nb_variables
    brute = inertie_brute(valeurs_propres, inertie_totale)
    benzecri = inertie_benzecri(valeurs_propres, nb_variables)
    nb_axes_utiles = int((valeurs_propres > 1 / nb_variables).sum())

    df_inertie = pd.DataFrame({
        'Axe': [f'Axe {i + 1}' for i in range(len(valeurs_propres))],
        'Valeur Propre': valeurs_propres,
        'λ > 1/p': np.where(valeurs_propres > 1 / nb_variables, 'oui', 'non'),
        'Inertie Brute (%)': brute,
        'Inertie Brute Cumulée (%)': np.cumsum(brute),
        'Inertie Benzécri (%)': benzecri,
        'Inertie Benzécri Cumulée (%)': np.cumsum(benzecri)
    })

    print(f"\n=== Inertie expliquée par les axes factoriels (p = {nb_variables} variables, 1/p = {1 / nb_variables:.3f}) ===")
    print(df_inertie.round(2).to_string(index=False))

    # ===== 5. COORDONNÉES + CONTRIBUTIONS DES MODALITÉS =====
    coordonnees = modele.column_coordinates(tableau_disjonctif).loc[tableau_disjonctif.columns].iloc[:, :n_axes_tableau]
    contributions = modele.column_contributions_.loc[tableau_disjonctif.columns].iloc[:, :n_axes_tableau]
    if np.isclose(contributions.sum().iloc[0], 1.0):
        contributions = contributions * 100

    seuil_contribution = facteur_seuil * 100 / nb_modalites
    effectifs = tableau_disjonctif.sum(axis=0)

    tableau_modalites = pd.DataFrame({
        'Variable': [variable_de(m) for m in tableau_disjonctif.columns],
        'Modalité': [m.split('__', 1)[1] for m in tableau_disjonctif.columns],
        'Effectif': effectifs.values.astype(int),
        'Fréquence (%)': effectifs.values / len(tableau_disjonctif) * 100
    }, index=tableau_disjonctif.columns)

    for i in range(n_axes_tableau):
        tableau_modalites[f'Coord Axe {i + 1}'] = coordonnees[i].values
        tableau_modalites[f'Contrib Axe {i + 1} (%)'] = contributions[i].values

    colonnes_coord = [f'Coord Axe {i + 1}' for i in range(n_axes_tableau)]
    colonnes_contrib = [f'Contrib Axe {i + 1} (%)' for i in range(n_axes_tableau)]

    tableau_modalites['Contrib max (%)'] = tableau_modalites[colonnes_contrib].max(axis=1)

    # ===== 6. DIAGNOSTIC PAR VARIABLE =====
    groupes = tableau_modalites.groupby('Variable', sort=False)
    resume_variables = pd.DataFrame({
        'Nb modalités': groupes.size(),
        'Nb modalités > seuil': groupes['Contrib max (%)'].apply(lambda s: int((s > seuil_contribution).sum())),
        'Contrib max (%)': groupes['Contrib max (%)'].max()
    })
    for i in range(n_axes_tableau):
        resume_variables[f'Max Axe {i + 1} (%)'] = groupes[colonnes_contrib[i]].max()

    resume_variables['Statut'] = np.where(resume_variables['Nb modalités > seuil'] == 0, 'À RETIRER ?', 'OK')
    resume_variables = resume_variables.sort_values('Contrib max (%)')
    variables_a_revoir = resume_variables.index[resume_variables['Statut'] != 'OK'].tolist()

    # ===== 7. MODALITÉS PROCHES =====
    lignes_proximite = []
    for variable, groupe in tableau_modalites.groupby('Variable', sort=False):
        enregistrements = groupe.reset_index(drop=True).to_dict('records')
        for a, b in combinations(enregistrements, 2):
            vecteur_a = np.array([a[c] for c in colonnes_coord[:n_axes_proximite]], dtype=float)
            vecteur_b = np.array([b[c] for c in colonnes_coord[:n_axes_proximite]], dtype=float)
            lignes_proximite.append({
                'Variable': variable,
                'Modalité A': a['Modalité'],
                'Modalité B': b['Modalité'],
                'Effectif A': a['Effectif'],
                'Effectif B': b['Effectif'],
                'Distance': np.linalg.norm(vecteur_a - vecteur_b)
            })

    proximites = pd.DataFrame(lignes_proximite).sort_values('Distance').reset_index(drop=True)
    proximites['Proches ?'] = np.where(proximites['Distance'] < seuil_proximite, 'OUI', '')

    # ===== 8. AFFICHAGE DES RÉSULTATS PAR AXE =====
    print(f"\nSeuil de contribution 100/K = {seuil_contribution:.2f} %")
    format_base = {'Fréquence (%)': '{:.1f}'}

    print("\n=== Coordonnées et contributions des modalités par axe ===")
    for i in range(n_axes_tableau):
        c_coord = f'Coord Axe {i + 1}'
        c_contrib = f'Contrib Axe {i + 1} (%)'
        cols_axe = ['Variable', 'Modalité', 'Effectif', 'Fréquence (%)', c_coord, c_contrib]
        df_axe = tableau_modalites[cols_axe].copy()
    
        format_axe = format_base.copy()
        format_axe[c_coord] = '{:.3f}'
        format_axe[c_contrib] = '{:.2f}'
    
        print(f"\n--- Axe {i + 1} ---")
        display(
            df_axe.style
            .apply(lambda df: surligner_modalites_par_axe(df, c_coord, c_contrib, seuil_contribution, variables_a_revoir), axis=None)
            .format(format_axe)
        )

    print("\n=== Diagnostic par variable ===")
    display(
        resume_variables.style.apply(lambda df: surligner_resume(df, seuil_contribution), axis=None)
        .format({c: '{:.2f}' for c in resume_variables.columns if '%' in c})
    )
    print(f"Variables à examiner pour la prochaine itération : {variables_a_revoir if variables_a_revoir else 'aucune'}")
    print(f"variables_a_retirer à copier = {sorted(set(variables_a_retirer + variables_a_revoir))}")

    print(f"\n=== Modalités proches (distance sur {n_axes_proximite} axes, seuil = {seuil_proximite}) ===")
    nb_proches = int((proximites['Proches ?'] == 'OUI').sum())
    print(f"{nb_proches} paire(s) proche(s) sur {len(proximites)} paires comparées")
    display(proximites.head(30).style.apply(surligner_proximite, axis=None).format({'Distance': '{:.3f}'}))

    # ===== 9. PLAN FACTORIEL (axes 1-2) =====
    contribue_plan = tableau_modalites[['Contrib Axe 1 (%)', 'Contrib Axe 2 (%)']].max(axis=1) > seuil_contribution
    plt.figure(figsize=(12, 8))
    plt.scatter(tableau_modalites.loc[~contribue_plan, 'Coord Axe 1'],
                tableau_modalites.loc[~contribue_plan, 'Coord Axe 2'],
                color='lightgrey', s=50, edgecolor='grey', label='Contribution <= seuil (axes 1-2)')
    plt.scatter(tableau_modalites.loc[contribue_plan, 'Coord Axe 1'],
                tableau_modalites.loc[contribue_plan, 'Coord Axe 2'],
                color='navy', s=90, edgecolor='black', label='Contribution > seuil (axes 1-2)')
    for nom, ligne in tableau_modalites.iterrows():
        plt.text(ligne['Coord Axe 1'] + 0.015, ligne['Coord Axe 2'] + 0.015, nom,
                 fontsize=8, fontweight='bold' if contribue_plan[nom] else 'normal',
                 alpha=0.9 if contribue_plan[nom] else 0.5)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title(f"ACM itération {numero_iteration} : plan factoriel (seuil {seuil_contribution:.2f} %)", fontsize=13)
    plt.xlabel(f"Axe 1 ({brute[0]:.2f} % brute | {benzecri[0]:.2f} % Benzécri)")
    plt.ylabel(f"Axe 2 ({brute[1]:.2f} % brute | {benzecri[1]:.2f} % Benzécri)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ===== 10. OBJETS POUR LA CAH =====
    modele_final = modele
    colonnes_finales = list(tableau_disjonctif.columns)
    variables_finales = sorted({variable_de(m) for m in colonnes_finales})
    coordonnees_individus = modele_final.row_coordinates(tableau_disjonctif).iloc[:, :n_axes_tableau]

    resultats_locaux = locals().copy()
    noms_resultats = (
        "variables_acm", "indices_echantillon", "echantillon_acm",
        "tableau_disjonctif", "modele", "valeurs_propres", "brute",
        "benzecri", "df_inertie", "tableau_modalites", "resume_variables",
        "variables_a_revoir", "proximites", "seuil_contribution",
    )
    return {nom: resultats_locaux[nom] for nom in noms_resultats}


def executer_iteration_2(donnees_nettoyees, variables_a_retirer, *, graine=42, numero_iteration=2, n_axes_tableau=5, n_axes_proximite=5, facteur_seuil=1.0, seuil_proximite=0.20, variables_exclues=None):
    """Reproduit les tableaux, diagnostics et graphiques de l’itération historique."""
    if variables_exclues is None:
        variables_exclues = ["id_anonyme", "statut_abonnement"]
    # ITERATION 2



    # ===== 1. ÉCHANTILLON DE 20 % (mêmes individus à chaque itération) =====
    variables_acm = [
        col for col in donnees_nettoyees.select_dtypes(include=['object', 'category']).columns
        if col not in variables_exclues + variables_a_retirer
    ]
    indices_echantillon = donnees_nettoyees.sample(frac=0.20, random_state=graine).index
    echantillon_acm = donnees_nettoyees.loc[indices_echantillon, variables_acm]
    print(f"Itération {numero_iteration} : {len(echantillon_acm)} individus, {len(variables_acm)} variables")

    # ===== 2. TABLEAU DISJONCTIF COMPLET =====
    tableau_disjonctif = pd.get_dummies(echantillon_acm, prefix_sep='__').astype(float)
    nb_modalites = tableau_disjonctif.shape[1]
    nb_variables = len(variables_acm)

    # Nombre maximal d'axes factoriels non triviaux
    nombre_axes_max = nb_modalites - nb_variables
    n_composantes = min(nombre_axes_max, len(echantillon_acm) - 1)

    print(f"Nombre de modalités actives (K) : {nb_modalites}")
    print(f"Nombre d'axes factoriels calculés : {n_composantes}")

    # ===== 3. AJUSTEMENT DE L'ACM (AFC sur le tableau disjonctif) =====
    modele = prince.CA(
        n_components=n_composantes,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=graine
    )
    modele = modele.fit(tableau_disjonctif)

    # ===== 4. TABLEAU D'INERTIE =====
    valeurs_propres = np.array(modele.eigenvalues_)
    inertie_totale = (nb_modalites - nb_variables) / nb_variables
    brute = inertie_brute(valeurs_propres, inertie_totale)
    benzecri = inertie_benzecri(valeurs_propres, nb_variables)
    nb_axes_utiles = int((valeurs_propres > 1 / nb_variables).sum())

    df_inertie = pd.DataFrame({
        'Axe': [f'Axe {i + 1}' for i in range(len(valeurs_propres))],
        'Valeur Propre': valeurs_propres,
        'λ > 1/p': np.where(valeurs_propres > 1 / nb_variables, 'oui', 'non'),
        'Inertie Brute (%)': brute,
        'Inertie Brute Cumulée (%)': np.cumsum(brute),
        'Inertie Benzécri (%)': benzecri,
        'Inertie Benzécri Cumulée (%)': np.cumsum(benzecri)
    })

    print(f"\n=== Inertie expliquée par les axes factoriels (p = {nb_variables} variables, 1/p = {1 / nb_variables:.3f}) ===")
    print(df_inertie.round(2).to_string(index=False))

    # ===== 5. COORDONNÉES + CONTRIBUTIONS DES MODALITÉS =====
    coordonnees = (
        modele
        .column_coordinates(tableau_disjonctif)
        .loc[tableau_disjonctif.columns]
        .iloc[:, :n_axes_tableau]
    )

    contributions = (
        modele
        .column_contributions_
        .loc[tableau_disjonctif.columns]
        .iloc[:, :n_axes_tableau]
    )

    if np.isclose(contributions.sum().iloc[0], 1.0):
        contributions = contributions * 100

    seuil_contribution = facteur_seuil * 100 / nb_modalites
    effectifs = tableau_disjonctif.sum(axis=0)

    tableau_modalites = pd.DataFrame({
        'Variable': [variable_de(m) for m in tableau_disjonctif.columns],
        'Modalité': [m.split('__', 1)[1] for m in tableau_disjonctif.columns],
        'Effectif': effectifs.values.astype(int),
        'Fréquence (%)': effectifs.values / len(tableau_disjonctif) * 100
    }, index=tableau_disjonctif.columns)

    for i in range(n_axes_tableau):
        tableau_modalites[f'Coord Axe {i + 1}'] = coordonnees[i].values
        tableau_modalites[f'Contrib Axe {i + 1} (%)'] = contributions[i].values

    colonnes_coord = [
        f'Coord Axe {i + 1}' for i in range(n_axes_tableau)
    ]

    colonnes_contrib = [
        f'Contrib Axe {i + 1} (%)' for i in range(n_axes_tableau)
    ]

    tableau_modalites['Contrib max (%)'] = (
        tableau_modalites[colonnes_contrib].max(axis=1)
    )

    # ===== 6. DIAGNOSTIC PAR VARIABLE =====
    groupes = tableau_modalites.groupby('Variable', sort=False)
    resume_variables = pd.DataFrame({
        'Nb modalités': groupes.size(),
        'Nb modalités > seuil': groupes['Contrib max (%)'].apply(lambda s: int((s > seuil_contribution).sum())),
        'Contrib max (%)': groupes['Contrib max (%)'].max()
    })
    for i in range(n_axes_tableau):
        resume_variables[f'Max Axe {i + 1} (%)'] = groupes[colonnes_contrib[i]].max()

    resume_variables['Statut'] = np.where(resume_variables['Nb modalités > seuil'] == 0, 'À RETIRER ?', 'OK')
    resume_variables = resume_variables.sort_values('Contrib max (%)')
    variables_a_revoir = resume_variables.index[resume_variables['Statut'] != 'OK'].tolist()

    # ===== 7. MODALITÉS PROCHES =====
    lignes_proximite = []
    for variable, groupe in tableau_modalites.groupby('Variable', sort=False):
        enregistrements = groupe.reset_index(drop=True).to_dict('records')
        for a, b in combinations(enregistrements, 2):
            vecteur_a = np.array([a[c] for c in colonnes_coord[:n_axes_proximite]], dtype=float)
            vecteur_b = np.array([b[c] for c in colonnes_coord[:n_axes_proximite]], dtype=float)
            lignes_proximite.append({
                'Variable': variable,
                'Modalité A': a['Modalité'],
                'Modalité B': b['Modalité'],
                'Effectif A': a['Effectif'],
                'Effectif B': b['Effectif'],
                'Distance': np.linalg.norm(vecteur_a - vecteur_b)
            })

    proximites = pd.DataFrame(lignes_proximite).sort_values('Distance').reset_index(drop=True)
    proximites['Proches ?'] = np.where(proximites['Distance'] < seuil_proximite, 'OUI', '')

    # ===== 8. AFFICHAGE DES RÉSULTATS PAR AXE =====
    print(f"\nSeuil de contribution 100/K = {seuil_contribution:.2f} %")
    format_base = {'Fréquence (%)': '{:.1f}'}

    print("\n=== Coordonnées et contributions des modalités par axe ===")
    for i in range(n_axes_tableau):
        c_coord = f'Coord Axe {i + 1}'
        c_contrib = f'Contrib Axe {i + 1} (%)'
        cols_axe = ['Variable', 'Modalité', 'Effectif', 'Fréquence (%)', c_coord, c_contrib]
        df_axe = tableau_modalites[cols_axe].copy()
    
        format_axe = format_base.copy()
        format_axe[c_coord] = '{:.3f}'
        format_axe[c_contrib] = '{:.2f}'
    
        print(f"\n--- Axe {i + 1} ---")
        display(
            df_axe.style
            .apply(lambda df: surligner_modalites_par_axe(df, c_coord, c_contrib, seuil_contribution, variables_a_revoir), axis=None)
            .format(format_axe)
        )

    print("\n=== Diagnostic par variable ===")
    display(
        resume_variables.style.apply(lambda df: surligner_resume(df, seuil_contribution), axis=None)
        .format({c: '{:.2f}' for c in resume_variables.columns if '%' in c})
    )
    print(f"Variables à examiner pour la prochaine itération : {variables_a_revoir if variables_a_revoir else 'aucune'}")
    print(f"variables_a_retirer à copier = {sorted(set(variables_a_retirer + variables_a_revoir))}")

    print(f"\n=== Modalités proches (distance sur {n_axes_proximite} axes, seuil = {seuil_proximite}) ===")
    nb_proches = int((proximites['Proches ?'] == 'OUI').sum())
    print(f"{nb_proches} paire(s) proche(s) sur {len(proximites)} paires comparées")
    display(proximites.head(30).style.apply(surligner_proximite, axis=None).format({'Distance': '{:.3f}'}))

    # ===== 9. PLAN FACTORIEL (axes 1-2) =====
    contribue_plan = tableau_modalites[['Contrib Axe 1 (%)', 'Contrib Axe 2 (%)']].max(axis=1) > seuil_contribution
    plt.figure(figsize=(12, 8))
    plt.scatter(tableau_modalites.loc[~contribue_plan, 'Coord Axe 1'],
                tableau_modalites.loc[~contribue_plan, 'Coord Axe 2'],
                color='lightgrey', s=50, edgecolor='grey', label='Contribution <= seuil (axes 1-2)')
    plt.scatter(tableau_modalites.loc[contribue_plan, 'Coord Axe 1'],
                tableau_modalites.loc[contribue_plan, 'Coord Axe 2'],
                color='navy', s=90, edgecolor='black', label='Contribution > seuil (axes 1-2)')
    for nom, ligne in tableau_modalites.iterrows():
        plt.text(ligne['Coord Axe 1'] + 0.015, ligne['Coord Axe 2'] + 0.015, nom,
                 fontsize=8, fontweight='bold' if contribue_plan[nom] else 'normal',
                 alpha=0.9 if contribue_plan[nom] else 0.5)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title(f"ACM itération {numero_iteration} : plan factoriel (seuil {seuil_contribution:.2f} %)", fontsize=13)
    plt.xlabel(f"Axe 1 ({brute[0]:.2f} % brute | {benzecri[0]:.2f} % Benzécri)")
    plt.ylabel(f"Axe 2 ({brute[1]:.2f} % brute | {benzecri[1]:.2f} % Benzécri)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

    resultats_locaux = locals().copy()
    noms_resultats = (
        "variables_acm", "indices_echantillon", "echantillon_acm",
        "tableau_disjonctif", "modele", "valeurs_propres", "brute",
        "benzecri", "df_inertie", "tableau_modalites", "resume_variables",
        "variables_a_revoir", "proximites", "seuil_contribution",
    )
    return {nom: resultats_locaux[nom] for nom in noms_resultats}
