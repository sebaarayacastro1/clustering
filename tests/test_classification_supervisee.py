"""Tests du contrôle des références et de l'apprentissage supervisé."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score

from src.classification_hierarchique import selectionner_sous_echantillon
from src.classification_supervisee import (
    charger_reference_cah, decouper_apprentissage, construire_modeles, evaluer_modele,
)


def creer_reference(racine: Path, graine: int = 42):
    (racine / "resultats/cah").mkdir(parents=True)
    (racine / "modeles").mkdir()
    positions = selectionner_sous_echantillon(80, 30, 42)
    pd.DataFrame(np.arange(80*5).reshape(80,5) / 100, columns=[f"Axe_{i}" for i in range(1,6)]).to_csv(
        racine / "resultats/coordonnees_factorielles.csv", index=False)
    pd.DataFrame({"position_echantillon": np.arange(80), "indice_ligne_source": np.arange(80)+1000}).to_csv(
        racine / "resultats/correspondance_clients_acm.csv", index=False)
    (racine / "modeles/acm_final_metadonnees.json").write_text(json.dumps({"iteration":3,"nombre_axes_calcules":5,
        "variables_actives":["a"]*10,"modalites_actives":["x"]*42}), encoding="utf-8")
    labels=np.tile([1,2,3,4,5],6)
    pd.DataFrame({"position_echantillon":positions, "indice_ligne_source":positions+1000,
                  "groupe_k5":labels}).to_csv(racine / "resultats/cah/partitions_essai_4axes.csv",index=False)
    (racine / "resultats/cah/parametres_essai_4axes.json").write_text(json.dumps({"mode":"essai", "nombre_individus":30,
        "nombre_axes":4,"k_central":5,"graine":graine,"methode":"ward","valeurs_k":[4,5,6]}),encoding="utf-8")
    return positions


def test_chargement_reference_valide(tmp_path):
    creer_reference(tmp_path)
    X,y,p,cols,meta,acm=charger_reference_cah(tmp_path,taille=30)
    assert X.shape == (30,4)
    assert y.nunique()==5
    assert np.array_equal(p["position_echantillon"],selectionner_sous_echantillon(80,30,42))


def test_refus_mauvaise_graine(tmp_path):
    creer_reference(tmp_path,graine=123)
    with pytest.raises(ValueError,match="graine"):
        charger_reference_cah(tmp_path,taille=30)


def test_refus_correspondance_incorrecte(tmp_path):
    creer_reference(tmp_path)
    f=tmp_path / "resultats/cah/partitions_essai_4axes.csv"
    p=pd.read_csv(f)
    p.loc[0,"indice_ligne_source"] = -100
    p.to_csv(f,index=False)
    with pytest.raises(ValueError,match="Correspondance"):
        charger_reference_cah(tmp_path,taille=30)


def test_refus_positions_melangees(tmp_path):
    creer_reference(tmp_path)
    f=tmp_path / "resultats/cah/partitions_essai_4axes.csv"
    p=pd.read_csv(f)
    p.loc[0,"position_echantillon"],p.loc[1,"position_echantillon"] = p.loc[1,"position_echantillon"],p.loc[0,"position_echantillon"]
    p.to_csv(f,index=False)
    with pytest.raises(ValueError,match="tirage"):
        charger_reference_cah(tmp_path,taille=30)


def test_decoupage_et_evaluation():
    Xv, yv=make_blobs(n_samples=250,centers=5,n_features=4,random_state=42,cluster_std=0.8)
    X=pd.DataFrame(Xv,columns=[f"Axe_{i}" for i in range(1,5)])
    y=pd.Series(yv+1)
    ia,it=decouper_apprentissage(X,y)
    assert len(ia)==200 and len(it)==50
    assert set(ia).isdisjoint(set(it))
    modele=construire_modeles()["regression_logistique"].fit(X.iloc[ia],y.iloc[ia])
    bilan,matrice,groupes,pred=evaluer_modele("regression_logistique",modele,X.iloc[it],y.iloc[it])
    assert 0 <= bilan["f1_macro"] <=1
    assert matrice.to_numpy().sum()==50
    assert len(groupes)==5 and len(pred)==50
