"""Fonctions de mise en évidence et de visualisation des résultats d'ACM."""
import pandas as pd

# Fonctions de mise en évidence
def surligner_modalites_par_axe(df, col_coord, col_contrib, seuil_contribution, variables_a_revoir):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    # Vert pastel doux avec texte foncé pour le contraste
    style_vert = 'background-color: #e2f0d9; color: #1b3a12; font-weight: bold;'
    styles.loc[df[col_contrib] > seuil_contribution, col_contrib] = style_vert
    styles.loc[df[col_contrib] > seuil_contribution, col_coord] = style_vert
    styles.loc[df['Variable'].isin(variables_a_revoir), ['Variable', 'Modalité']] = 'background-color: #fce4d6; color: #5c1d06;'
    return styles

def surligner_resume(df, seuil_contribution):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    styles.loc[df['Statut'] != 'OK', :] = 'background-color: #fce4d6; color: #5c1d06;'
    for c in [col for col in df.columns if col.startswith('Max Axe')]:
        styles.loc[df[c] > seuil_contribution, c] = 'background-color: #e2f0d9; color: #1b3a12; font-weight: bold;'
    return styles

def surligner_proximite(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    styles.loc[df['Proches ?'] == 'OUI', :] = 'background-color: #fff2cc; color: #594500;'
    return styles