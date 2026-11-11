"""Adaptation des modalités ACM finales pour le formulaire client.

La représentation exposée à l'utilisateur est une modalité finale de l'ACM.
Pour ne pas changer le contrat du moteur existant, le formulaire restitue
une valeur source représentative dont la recodification est exactement égale
à cette modalité finale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


LIBELLES_CHAMPS = {
    "anciennete_client_rec": "Depuis combien de temps le client utilise-t-il Karafun ?",
    "anciennete_abonnement_rec": "Depuis combien de temps le client est-il abonné ?",
    "methode_paiement_rec": "Quel est son mode de paiement ?",
    "valeur_12M_rec": "Quelle est la valeur du client sur les 12 derniers mois ?",
    "avg_sessions_actifs_rec": "Quelle est sa fréquence moyenne de sessions pendant les périodes actives ?",
    "avg_songs_sessions_rec": "Combien de chansons écoute-t-il en moyenne par session ?",
    "type_client": "Quel est le statut actuel du client ?",
    "ratio_fidelite_tr": "Quel est le niveau de fidélité du client ?",
    "contactable": "Le client peut-il être contacté ?",
    "utilise_remote": "Le client utilise-t-il la fonctionnalité « Remote » ?",
    "utilise_remote_rec": "Le client utilise-t-il la fonctionnalité « Remote » ?",
}

LIBELLES_MODALITES = {
    "web_et_direct": "Paiement sur le Web ou directement",
    "telephone": "Paiement sur téléphone (Apple / Android)",
    "Amazon": "Amazon",
    "Credit": "Crédit",
    "00_Inactif": "Aucune activité observée",
    "03_10_et_plus": "10 chansons ou plus par session",
    "oui": "Oui",
    "non": "Non",
    "nouveau": "Nouveau client",
    "churn": "Client en churn",
    "reactivated": "Client réactivé",
}


@dataclass(frozen=True)
class OptionModalite:
    """Une modalité ACM finale et sa valeur source représentative."""

    code_final: str
    valeur_source: str
    libelle: str


def libelle_modalite(code_final: str) -> str:
    """Retire les codes de tri et affiche un texte lisible en français."""
    if code_final in LIBELLES_MODALITES:
        return LIBELLES_MODALITES[code_final]
    propre = re.sub(r"^\d+_", "", str(code_final)).replace("_", " ")
    propre = re.sub(r"\s+", " ", propre).strip()
    propre = re.sub(r"(\d+)\s*a\s*(\d+)", r"\1 à \2", propre)
    return propre[:1].upper() + propre[1:]


def libelle_variable(variable: str) -> str:
    """Question métier pour les variables connues, sinon libellé explicatif."""
    if variable in LIBELLES_CHAMPS:
        return LIBELLES_CHAMPS[variable]
    nom = re.sub(r"_rec$", "", variable).replace("_", " ").strip()
    return f"Quelle est la catégorie correspondant à « {nom} » ?"


def _modalite_finale(predicteur: Any, variable: str, valeur_source: str) -> str:
    if variable in predicteur.regroupements:
        modalite = predicteur.regroupements[variable]["correspondances"][valeur_source]
    else:
        modalite = valeur_source
    for nouvelle, anciennes in predicteur.fusions.get(variable, {}).items():
        if modalite in anciennes:
            return nouvelle
    return modalite


def options_modalites_finales(predicteur: Any, variable: str) -> list[OptionModalite]:
    """Construit UNE option par modalité ACM réellement active.

    Ne présente jamais des catégories sources comme Apple/Android ou des
    anciens intervalles de durée. Vérifie que chaque modalité active peut
    être convertie en entrée valide par le moteur individuel existant.
    """
    if variable not in predicteur.modalites_par_variable:
        raise ValueError(f"Variable absente des métadonnées ACM : {variable}")
    actives = set(predicteur.modalites_par_variable[variable])
    if variable in predicteur.regroupements:
        sources = predicteur.regroupements[variable]["correspondances"].keys()
    else:
        sources = sorted(actives)

    representatives: dict[str, str] = {}
    for source in sources:
        source = str(source)
        if source == "Others" and predicteur.sources_requises[variable] == "methode_paiement":
            continue
        finale = _modalite_finale(predicteur, variable, source)
        if finale in actives:
            representatives.setdefault(finale, source)

    non_representees = actives.difference(representatives)
    if non_representees:
        raise ValueError(
            f"Modalités ACM non représentées pour {variable} : {sorted(non_representees)}"
        )
    options = []
    for finale in sorted(actives):
        libelle = libelle_modalite(finale)
        if variable == "ratio_fidelite_tr":
            libelle = f"{libelle} %"
        options.append(OptionModalite(finale, representatives[finale], libelle))
    return options
