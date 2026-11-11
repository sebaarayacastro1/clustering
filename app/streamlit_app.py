"""Interface Streamlit de prédiction individuelle des segments Karafun.

Les modalités visibles sont les catégories ACTIVES FINALES de l'ACM 3.
La saisie est adaptée aux valeurs sources requises par le moteur existant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from app.modalites_interface import (  # noqa: E402
    libelle_variable,
    options_modalites_finales,
)
from src.prediction_client import PredicteurClient  # noqa: E402


st.set_page_config(page_title="Karafun | Segmentation clients", page_icon="🎤", layout="wide")


@st.cache_resource(show_spinner=False)
def charger_predicteur(racine: str) -> PredicteurClient:
    """Réutilise les artefacts enregistrés pour la prédiction."""
    return PredicteurClient(Path(racine))


def afficher_resultat(resultat) -> None:
    """Affiche le segment et un tableau lisible des cinq scores."""
    st.success(f"Segment attribué : G{resultat.groupe_k5} — {resultat.nom_segment}")
    tableau = pd.DataFrame(
        [
            {"Segment": groupe, "Score estimé (%)": round(score * 100, 2)}
            for groupe, score in resultat.probabilites.items()
        ]
    )
    st.subheader("Scores estimés par segment")
    st.dataframe(tableau, hide_index=True, use_container_width=True)
    st.caption(
        "Les scores sont des estimations du classifieur et ne sont pas calibrés. "
        "Ils ne correspondent pas à des probabilités de churn futur."
    )


def main() -> None:
    st.title("Karafun — Segmentation des clients")
    st.write("Renseignez le profil d'un client pour identifier son segment.")

    try:
        predicteur = charger_predicteur(str(RACINE))
    except (FileNotFoundError, ValueError, KeyError, ImportError) as erreur:
        st.error(f"Impossible de charger les éléments nécessaires : {erreur}")
        st.stop()

    st.subheader("Profil du client")
    st.caption("Choisissez une réponse par question. Tous les champs sont obligatoires.")

    valeurs_source = {}
    options_retenues = {}
    with st.form("prediction_individuelle", clear_on_submit=False):
        colonnes = st.columns(2)
        for position, variable in enumerate(predicteur.variables_actives):
            source = predicteur.sources_requises[variable]
            try:
                choix = options_modalites_finales(predicteur, variable)
            except (ValueError, KeyError) as erreur:
                st.error(f"Configuration des modalités incorrecte : {erreur}")
                st.stop()
            with colonnes[position % 2]:
                selection = st.selectbox(
                    libelle_variable(variable),
                    options=choix,
                    index=None,
                    placeholder="Sélectionner une réponse…",
                    format_func=lambda option: option.libelle,
                    key=f"modalite_{variable}",
                )
                options_retenues[variable] = selection
                if selection is not None:
                    valeurs_source[source] = selection.valeur_source
        envoyer = st.form_submit_button("Identifier le segment", type="primary", use_container_width=True)

    if envoyer:
        non_renseignees = [
            libelle_variable(variable)
            for variable, selection in options_retenues.items()
            if selection is None
        ]
        if non_renseignees:
            st.warning("Veuillez renseigner : " + " ; ".join(non_renseignees))
            return
        try:
            resultat = predicteur.predire(valeurs_source)
        except (ValueError, KeyError) as erreur:
            st.error(f"Prédiction impossible : {erreur}")
            return
        afficher_resultat(resultat)

    st.divider()
    st.caption(
        "Prototype de consultation : les informations saisies ne sont pas enregistrées."
    )


if __name__ == "__main__":
    main()
