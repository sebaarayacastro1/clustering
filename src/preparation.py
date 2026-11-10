"""Préparation des variables et application des regroupements historiques."""
from pathlib import Path
import pandas as pd
import yaml

VARIABLES_A_SUPPRIMER = [
    'id_anonyme', 'anciennete_client_tr', 'anciennete_abonnement_tr',
    'methode_paiement', 'valeur_12M_tr', 'avg_sessions_actifs_tr',
    'avg_songs_sessions_tr', 'zone_geo', 'type_very_first_abo',
    'type_first_abo_last_streak',
]

def charger_regroupements(chemin):
    """Lit les correspondances de recodification du fichier YAML."""
    with open(chemin, encoding='utf-8') as fichier:
        return yaml.safe_load(fichier)['regroupements']

def preparer_donnees(donnees, regroupements, verifier_modalites=True):
    """Reproduit les recodifications et exclusions du notebook initial."""
    donnees = donnees.copy()
    for variable_cible, definition in regroupements.items():
        variable_source = definition['source']
        if variable_source not in donnees:
            raise KeyError(f"Variable source absente : {variable_source}")
        donnees[variable_cible] = donnees[variable_source].map(definition['correspondances'])
        if verifier_modalites:
            inconnues = donnees.loc[donnees[variable_source].notna() & donnees[variable_cible].isna(),variable_source].unique().tolist()
            if inconnues:
                raise ValueError(f"Modalités non recodées dans {variable_source} : {inconnues}")
    # Reprise exacte du filtrage préalable à la création des données nettoyées.
    donnees = donnees.loc[donnees['methode_paiement'] != 'Others'].copy()
    donnees_nettoyees = donnees.drop(columns=VARIABLES_A_SUPPRIMER)
    return donnees, donnees_nettoyees

def decrire_distributions(donnees):
    """Construit les tableaux d'effectifs des modalités recodées."""
    return {nom: pd.DataFrame({'Fréquence': donnees[nom].value_counts(),
                               'Pourcentage (%)': (donnees[nom].value_counts(normalize=True)*100).round(2)})
            for nom in donnees.columns if nom.endswith('_rec')}
