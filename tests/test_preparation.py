"""Vérifications des règles de regroupement reproduites."""
from pathlib import Path
import pandas as pd
from src.preparation import charger_regroupements, preparer_donnees

def test_regroupements_charges():
    regles = charger_regroupements(Path(__file__).resolve().parents[1] / 'config/regroupements.yaml')
    assert len(regles) == 7
    assert regles['methode_paiement_rec']['correspondances']['Stripe'] == 'web_et_direct'

def test_filtrage_others_et_recodification():
    regles = charger_regroupements(Path(__file__).resolve().parents[1] / 'config/regroupements.yaml')
    colonnes = {definition['source']: next(iter(definition['correspondances'])) for definition in regles.values()}
    colonnes.update({'id_anonyme':'1','type_very_first_abo':'X','type_first_abo_last_streak':'X'})
    donnees = pd.DataFrame([colonnes])
    donnees['methode_paiement'] = 'Stripe'
    _, nettoyees = preparer_donnees(donnees, regles)
    assert nettoyees['methode_paiement_rec'].iloc[0] == 'web_et_direct'
    assert 'id_anonyme' not in nettoyees.columns
