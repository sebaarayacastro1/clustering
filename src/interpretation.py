"""Outils de description des groupes une fois la CAH réalisée."""
import pandas as pd

def effectifs_groupes(etiquettes):
    """Retourne les effectifs de chaque groupe disponible."""
    return pd.Series(etiquettes, name='Groupe').value_counts().sort_index()
