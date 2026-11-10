"""Tests de conformité des profils CAH aux variables de l'ACM 3."""
from pathlib import Path
import json
import sys
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.interpretation import preparer_variables_acm_finales


@pytest.fixture
def environnement(tmp_path):
    chemin_config = tmp_path / 'config' / 'regroupements.yaml'
    chemin_config.parent.mkdir(parents=True)
    configuration = yaml.safe_load((Path(__file__).resolve().parents[2] / 'regroupements.yaml').read_text(encoding='utf-8'))
    chemin_config.write_text(yaml.safe_dump(configuration, allow_unicode=True), encoding='utf-8')
    variables = [
        'anciennete_client_rec', 'anciennete_abonnement_rec', 'methode_paiement_rec',
        'valeur_12M_rec', 'avg_sessions_actifs_rec', 'avg_songs_sessions_rec',
        'type_client', 'ratio_fidelite_tr', 'utilise_remote', 'contactable',
    ]
    donnees = pd.DataFrame({
        'anciennete_client_tr': ['00_moins_1_mois', '18_85a120_mois'],
        'anciennete_abonnement_tr': ['00_moins_1_mois', '19_121_mois_et_plus'],
        'methode_paiement': ['Stripe', 'Paypal'],
        'valeur_12M_tr': ['00_moins_10', '11_110_et_plus'],
        'avg_sessions_actifs_tr': ['00_pas_de_session', '07_6a8'],
        'avg_songs_sessions_tr': ['08_10a15', '12_50_et_plus'],
        'zone_geo': ['FR', 'US'],
        'type_client': ['A', 'B'],
        'ratio_fidelite_tr': ['faible', 'fort'],
        'utilise_remote': ['oui', 'non'],
        'contactable': ['oui', 'non'],
        'type_very_first_abo': ['X', 'X'],
        'type_first_abo_last_streak': ['X', 'X'],
        'id_anonyme': ['x', 'y'],
    })
    base_modalites = [
        'anciennete_client_rec__01_Moins_de_6_mois',
        'anciennete_client_rec__05_Plus_de_4_ans',
        'anciennete_abonnement_rec__01_Moins_de_3_mois',
        'anciennete_abonnement_rec__05_Plus_de_2_ans',
        'methode_paiement_rec__web_et_direct',
        'valeur_12M_rec__01_Moins_de_15',
        'valeur_12M_rec__05_Plus_de_90',
        'avg_sessions_actifs_rec__00_Inactif',
        'avg_sessions_actifs_rec__03_Eleve_4_a_8',
        'avg_songs_sessions_rec__03_10_et_plus',
        'type_client__A', 'type_client__B',
        'ratio_fidelite_tr__faible', 'ratio_fidelite_tr__fort',
        'utilise_remote__oui', 'utilise_remote__non',
        'contactable__oui', 'contactable__non',
    ]
    modalites = base_modalites + [f'type_client__modalite_absente_{i}' for i in range(42-len(base_modalites))]
    chemin_meta = tmp_path / 'modeles' / 'acm_final_metadonnees.json'
    chemin_meta.parent.mkdir(parents=True)
    chemin_meta.write_text(json.dumps({
        'iteration': 3, 'variables_actives': variables,
        'modalites_actives': modalites,
        'regroupements_acm_finaux': configuration['regroupements_acm_finaux'],
    }), encoding='utf-8')
    return tmp_path, donnees, chemin_meta


def test_reconstruit_dix_variables_et_fusion_finale(environnement):
    racine, donnees, _ = environnement
    acm, meta = preparer_variables_acm_finales(donnees, racine)
    assert acm.columns.tolist() == meta['variables_actives']
    assert acm.shape == (2, 10)
    assert acm['avg_songs_sessions_rec'].tolist() == ['03_10_et_plus', '03_10_et_plus']
    assert 'zone_geo_rec' not in acm
    assert 'id_anonyme' not in acm
    assert 'nb_abo_periode' not in acm


def test_refuse_une_modalite_inconnue(environnement):
    racine, donnees, _ = environnement
    donnees.loc[0, 'type_client'] = 'inconnu'
    with pytest.raises(ValueError, match='Modalités incompatibles'):
        preparer_variables_acm_finales(donnees, racine)


def test_refuse_configuration_divergente(environnement):
    racine, donnees, chemin_meta = environnement
    meta = json.loads(chemin_meta.read_text())
    meta['regroupements_acm_finaux'] = {}
    chemin_meta.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='divergent'):
        preparer_variables_acm_finales(donnees, racine)


def test_refuse_individu_exclu(environnement):
    racine, donnees, _ = environnement
    donnees.loc[0, 'methode_paiement'] = 'Others'
    with pytest.raises(ValueError, match='exclus'):
        preparer_variables_acm_finales(donnees, racine)
