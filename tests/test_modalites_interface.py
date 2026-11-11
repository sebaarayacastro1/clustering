"""Contrôles du formulaire final ACM, indépendants de Streamlit."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.modalites_interface import (
    libelle_modalite,
    libelle_variable,
    options_modalites_finales,
)


def _faux_predicteur(variable, mapping, finales, fusions=None, source="champ_source"):
    return SimpleNamespace(
        regroupements={variable: {"source": source, "correspondances": mapping}},
        fusions={variable: fusions or {}},
        modalites_par_variable={variable: set(finales)},
        sources_requises={variable: source},
    )


def test_paiement_apple_android_meme_modalite():
    pred = _faux_predicteur(
        "methode_paiement_rec",
        {"Apple": "telephone", "Android": "telephone", "Stripe": "web_et_direct",
         "Paypal": "web_et_direct", "Amazon": "Amazon", "Credit": "Credit", "Others": "Others"},
        {"telephone", "web_et_direct", "Amazon", "Credit"},
        source="methode_paiement",
    )
    options = options_modalites_finales(pred, "methode_paiement_rec")
    assert {o.code_final for o in options} == {"telephone", "web_et_direct", "Amazon", "Credit"}
    assert len(options) == 4
    assert "Apple" not in [o.libelle for o in options]
    assert "Android" not in [o.libelle for o in options]
    assert "Others" not in [o.code_final for o in options]
    for option in options:
        assert pred.regroupements["methode_paiement_rec"]["correspondances"][option.valeur_source] == option.code_final


def test_anciennete_abonnement_plus_de_deux_ans():
    mapping = {"13_25a30_mois": "05_Plus_de_2_ans", "19_121_mois_et_plus": "05_Plus_de_2_ans",
               "00_moins_1_mois": "01_Moins_de_3_mois"}
    pred = _faux_predicteur("anciennete_abonnement_rec", mapping,
                           {"05_Plus_de_2_ans", "01_Moins_de_3_mois"})
    options = options_modalites_finales(pred, "anciennete_abonnement_rec")
    assert [o.libelle for o in options] == ["Moins de 3 mois", "Plus de 2 ans"]
    assert all("19 121" not in o.libelle for o in options)


def test_fusion_acm_definitive():
    pred = _faux_predicteur("avg_songs_sessions_rec",
                           {"08_10a15": "03_Eleve_10_a_20", "10_20a30": "04_Tres_eleve_20_et_plus"},
                           {"03_10_et_plus"},
                           fusions={"03_10_et_plus": ["03_Eleve_10_a_20", "04_Tres_eleve_20_et_plus"]})
    options = options_modalites_finales(pred, "avg_songs_sessions_rec")
    assert len(options) == 1
    assert options[0].code_final == "03_10_et_plus"
    assert options[0].libelle == "10 chansons ou plus par session"


def test_modalite_non_representable_refusee():
    pred = _faux_predicteur("variable_rec", {"source": "01_Categorie"},
                           {"01_Categorie", "02_Autre"})
    with pytest.raises(ValueError, match="non représentées"):
        options_modalites_finales(pred, "variable_rec")


def test_libelles_metier():
    assert "Remote" in libelle_variable("utilise_remote")
    assert "fidélité" in libelle_variable("ratio_fidelite_tr")
    assert libelle_modalite("oui") == "Oui"
    assert libelle_modalite("reactivated") == "Client réactivé"


def test_yaml_reel_si_disponible():
    config = Path(__file__).resolve().parents[1] / "config" / "regroupements.yaml"
    if not config.is_file():
        pytest.skip("Configuration du projet non disponible dans cet environnement")
    regroupements = yaml.safe_load(config.read_text(encoding="utf-8"))["regroupements"]
    assert regroupements["anciennete_client_rec"]["correspondances"]["19_121_mois_et_plus"] == "05_Plus_de_4_ans"
    assert regroupements["anciennete_abonnement_rec"]["correspondances"]["19_121_mois_et_plus"] == "05_Plus_de_2_ans"
