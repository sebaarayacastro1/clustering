"""Prédiction individuelle des cinq segments CAH, sans réentraînement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml

from src.classification_supervisee import NOMS_METIER_K5
from src.extrapolation_clusters import preparer_tableau_disjonctif


@dataclass(frozen=True)
class ResultatPrediction:
    """Affectation d'un client et scores produits par le classifieur."""

    groupe_k5: int
    nom_segment: str
    probabilite_max: float
    probabilites: dict[str, float]


class PredicteurClient:
    """Charge les artefacts figés puis prédit un client à la fois."""

    def __init__(self, racine: str | Path):
        self.racine = Path(racine)
        chemins = {
            "acm": self.racine / "modeles" / "acm_final.joblib",
            "meta_acm": self.racine / "modeles" / "acm_final_metadonnees.json",
            "classification": self.racine / "modeles" / "classifieur_clusters_k5.joblib",
            "meta_classification": self.racine / "modeles" / "classifieur_clusters_k5_metadonnees.json",
            "config": self.racine / "config" / "regroupements.yaml",
        }
        manquants = [str(chemin) for chemin in chemins.values() if not chemin.is_file()]
        if manquants:
            raise FileNotFoundError("Artefacts manquants : " + ", ".join(manquants))

        with chemins["meta_acm"].open(encoding="utf-8") as fichier:
            self.meta_acm = json.load(fichier)
        with chemins["meta_classification"].open(encoding="utf-8") as fichier:
            self.meta_classification = json.load(fichier)
        with chemins["config"].open(encoding="utf-8") as fichier:
            config = yaml.safe_load(fichier)
        self.regroupements = config["regroupements"]
        self.acm = joblib.load(chemins["acm"])
        self.classifieur = joblib.load(chemins["classification"])

        self.variables_actives = list(self.meta_acm["variables_actives"])
        self.modalites_actives = list(self.meta_acm["modalites_actives"])
        self.colonnes_axes = list(self.meta_classification["colonnes_axes"])
        self.fusions = self.meta_acm["regroupements_acm_finaux"]
        classes = np.asarray(self.classifieur.classes_, dtype=int)
        if (
            self.meta_acm.get("iteration") != 3
            or len(self.variables_actives) != 10
            or len(self.modalites_actives) != 42
            or self.meta_classification.get("modele_selectionne") != "regression_logistique_multinomiale"
            or self.meta_classification.get("nombre_axes") != 4
            or self.meta_classification.get("nombre_groupes") != 5
            or len(self.colonnes_axes) != 4
            or not np.array_equal(classes, np.arange(1, 6))
        ):
            raise ValueError("Les artefacts ne correspondent pas à l'ACM 3 et au classifieur G1–G5.")
        self.classes = classes
        self.sources_requises = {
            variable: self.regroupements[variable]["source"] if variable in self.regroupements else variable
            for variable in self.variables_actives
        }
        if len(set(self.sources_requises.values())) != len(self.variables_actives):
            raise ValueError("Les variables sources ne sont pas distinctes.")
        # Liste des valeurs effectivement connues par l'ACM, par variable active.
        self.modalites_par_variable = {
            variable: {
                modalite[len(variable) + 2:]
                for modalite in self.modalites_actives
                if modalite.startswith(variable + "__")
            }
            for variable in self.variables_actives
        }
        if any(not valeurs for valeurs in self.modalites_par_variable.values()):
            raise ValueError("Modalités actives incomplètes dans les métadonnées ACM.")

    @property
    def champs_requis(self) -> list[str]:
        """Champs originaux nécessaires, indépendamment du CSV historique."""
        return list(dict.fromkeys([*self.sources_requises.values(), "methode_paiement"]))

    def preparer_client(self, client: Mapping[str, Any]) -> pd.DataFrame:
        """Recode un client avec les règles figées de l'ACM 3."""
        manquants = [champ for champ in self.champs_requis if champ not in client]
        if manquants:
            raise ValueError("Champs manquants : " + ", ".join(manquants))
        if client["methode_paiement"] == "Others":
            raise ValueError("Client non éligible : méthode de paiement 'Others'.")

        valeurs = {}
        for variable, source in self.sources_requises.items():
            brut = client[source]
            if pd.isna(brut) or (isinstance(brut, str) and not brut.strip()):
                raise ValueError(f"Valeur manquante pour {source}.")
            if variable in self.regroupements:
                correspondances = self.regroupements[variable]["correspondances"]
                if brut not in correspondances:
                    raise ValueError(f"Modalité inconnue pour {source} : {brut!r}.")
                valeur = correspondances[brut]
            else:
                valeur = str(brut)

            # Contrairement au regroupement historique par lot, cette opération
            # ne demande pas la présence des anciennes modalités dans le lot.
            for nouvelle, anciennes in self.fusions.get(variable, {}).items():
                if valeur in anciennes:
                    valeur = nouvelle
                    break
            if valeur not in self.modalites_par_variable[variable]:
                raise ValueError(f"Modalité absente de l'ACM pour {variable} : {valeur!r}.")
            valeurs[variable] = valeur
        return pd.DataFrame([valeurs], columns=self.variables_actives)

    def predire(self, client: Mapping[str, Any]) -> ResultatPrediction:
        """Applique la projection ACM et la prédiction, uniquement en lecture."""
        donnees = self.preparer_client(client)
        disjonctif = preparer_tableau_disjonctif(
            donnees, self.variables_actives, self.modalites_actives
        )
        coordonnees = self.acm.row_coordinates(disjonctif).iloc[:, :4].copy()
        coordonnees.columns = self.colonnes_axes
        if coordonnees.shape != (1, 4) or not np.isfinite(coordonnees.to_numpy(dtype=float)).all():
            raise ValueError("Projection ACM individuelle invalide.")

        groupe = int(self.classifieur.predict(coordonnees)[0])
        scores = np.asarray(self.classifieur.predict_proba(coordonnees), dtype=float)
        if (
            scores.shape != (1, 5)
            or not np.isfinite(scores).all()
            or np.any(scores < 0)
            or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-8)
            or groupe != int(self.classes[scores[0].argmax()])
        ):
            raise ValueError("Prédictions ou probabilités incohérentes.")
        return ResultatPrediction(
            groupe_k5=groupe,
            nom_segment=NOMS_METIER_K5[groupe],
            probabilite_max=float(scores[0].max()),
            probabilites={f"G{int(classe)}": float(score) for classe, score in zip(self.classes, scores[0])},
        )
