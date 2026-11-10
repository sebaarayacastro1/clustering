"""Interprétation descriptive et comparaisons métier des partitions CAH."""
from __future__ import annotations
from pandas.api.types import is_numeric_dtype

from pathlib import Path
import json
import numpy as np
import pandas as pd


def charger_partitions_et_donnees(
    racine: str | Path, mode: str = "essai", nombre_axes: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Charge les partitions et les individus sources avec une correspondance vérifiée."""
    racine = Path(racine)
    repertoire = racine / "resultats" / "cah"
    chemin_partitions = repertoire / f"partitions_{mode}_{nombre_axes}axes.csv"
    chemin_parametres = repertoire / f"parametres_{mode}_{nombre_axes}axes.json"
    chemin_source = racine / "data" / "brutes" / "td_clustering_karafun.csv"
    for chemin in [chemin_partitions, chemin_parametres, chemin_source]:
        if not chemin.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    partitions = pd.read_csv(chemin_partitions)
    with chemin_parametres.open(encoding="utf-8") as fichier:
        parametres = json.load(fichier)
    source = pd.read_csv(chemin_source)
    if partitions.empty or len(partitions) != int(parametres["nombre_individus"]):
        raise ValueError("Le nombre d'individus diffère des paramètres de la CAH.")
    if int(parametres["nombre_axes"]) != nombre_axes or parametres["mode"] != mode:
        raise ValueError("Les paramètres ne correspondent pas au mode demandé.")
    if "position_echantillon" in partitions:
        positions = partitions["position_echantillon"]
        if positions.isna().any() or positions.duplicated().any():
            raise ValueError("Positions d'échantillon invalides ou dupliquées.")

    # L'index technique n'est fiable que tant que le CSV et son ordre restent inchangés.
    if "indice_ligne_source" in partitions.columns:
        indices = pd.to_numeric(partitions["indice_ligne_source"], errors="raise")
        if indices.isna().any() or (indices % 1 != 0).any():
            raise ValueError("Indices de lignes sources invalides.")
        indices = indices.to_numpy(dtype=int)
        if ((indices < 0) | (indices >= len(source))).any() or len(set(indices)) != len(indices):
            raise ValueError("Indices sources hors bornes ou dupliqués.")
        descriptifs = source.iloc[indices].reset_index(drop=True)
        if "id_anonyme" in partitions and "id_anonyme" in descriptifs:
            if not descriptifs["id_anonyme"].reset_index(drop=True).equals(partitions["id_anonyme"]):
                raise ValueError("Identifiants incompatibles avec les positions sources.")
    elif "id_anonyme" in partitions.columns and "id_anonyme" in source.columns:
        if partitions["id_anonyme"].isna().any() or partitions["id_anonyme"].duplicated().any():
            raise ValueError("Identifiants de partitions manquants ou dupliqués.")
        if source["id_anonyme"].isna().any() or source["id_anonyme"].duplicated().any():
            raise ValueError("Identifiants de source manquants ou dupliqués.")
        index_source = source.set_index("id_anonyme", drop=False)
        absents = ~partitions["id_anonyme"].isin(index_source.index)
        if absents.any():
            raise ValueError("Certains identifiants sont absents de la base brute.")
        descriptifs = index_source.loc[partitions["id_anonyme"].to_numpy()].reset_index(drop=True)
        if not np.array_equal(descriptifs["id_anonyme"].to_numpy(), partitions["id_anonyme"].to_numpy()):
            raise AssertionError("Correspondance des individus non valide.")
    else:
        raise ValueError("Aucun identifiant fiable pour rapprocher les partitions des données brutes.")
    if len(descriptifs) != len(partitions):
        raise AssertionError("Taille des données descriptives différente de celle des partitions.")
    chemin_correspondance = racine / "resultats" / "correspondance_clients_acm.csv"
    if not chemin_correspondance.is_file():
        raise FileNotFoundError(
            f"Correspondance de l'ACM final introuvable : {chemin_correspondance}"
        )
    correspondance = pd.read_csv(chemin_correspondance)
    if "position_echantillon" not in partitions or "position_echantillon" not in correspondance:
        raise ValueError("Positions de l'échantillon ACM manquantes.")
    if correspondance["position_echantillon"].duplicated().any():
        raise ValueError("Positions ACM dupliquées dans le fichier de correspondance.")
    positions = partitions["position_echantillon"]
    if positions.isna().any() or (positions % 1 != 0).any():
        raise ValueError("Positions CAH non entières.")
    reference = correspondance.set_index("position_echantillon").reindex(positions.to_numpy())
    if reference.isna().all(axis=1).any():
        raise ValueError("Certaines positions de la CAH sont absentes de l'ACM.")
    for cle in ("indice_ligne_source", "id_anonyme"):
        if cle in partitions and cle in reference:
            attendu = reference[cle].reset_index(drop=True)
            obtenu = partitions[cle].reset_index(drop=True)
            if not attendu.equals(obtenu):
                raise ValueError(f"Discordance entre la CAH et l'ACM pour {cle}.")
    return partitions, descriptifs, parametres


def preparer_variables_acm_finales(
    donnees_brutes: pd.DataFrame, racine: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruit exactement les variables actives et les fusions de l’ACM 3.

    La liste des variables et des modalités provient de l'export de l'ACM ;
    aucune variable descriptive supplémentaire n'est sélectionnée.
    """
    from src.preparation import charger_regroupements, preparer_donnees
    import yaml

    racine = Path(racine)
    chemin_config = racine / "config" / "regroupements.yaml"
    chemin_metadonnees = racine / "modeles" / "acm_final_metadonnees.json"
    for chemin in (chemin_config, chemin_metadonnees):
        if not chemin.is_file():
            raise FileNotFoundError(f"Fichier de référence ACM introuvable : {chemin}")

    with chemin_metadonnees.open(encoding="utf-8") as fichier:
        metadonnees = json.load(fichier)
    if metadonnees.get("iteration") != 3:
        raise ValueError("Les métadonnées ne correspondent pas à l'ACM 3.")
    variables = metadonnees.get("variables_actives")
    modalites = metadonnees.get("modalites_actives")
    if not isinstance(variables, list) or len(variables) != 10 or len(set(variables)) != 10:
        raise ValueError("L'ACM 3 doit déclarer exactement dix variables actives distinctes.")
    if not isinstance(modalites, list) or len(modalites) != 42 or len(set(modalites)) != 42:
        raise ValueError("Les métadonnées de l'ACM 3 doivent déclarer 42 modalités actives distinctes.")

    with chemin_config.open(encoding="utf-8") as fichier:
        configuration = yaml.safe_load(fichier)
    fusions = configuration.get("regroupements_acm_finaux")
    fusions_reference = metadonnees.get("regroupements_acm_finaux")
    if not isinstance(fusions, dict) or not fusions:
        raise ValueError("Les fusions finales de l'ACM sont absentes du YAML.")
    if fusions != fusions_reference:
        raise ValueError("Le YAML et les métadonnées de l'ACM divergent sur les fusions finales.")

    regroupements = charger_regroupements(chemin_config)
    _, donnees_nettoyees = preparer_donnees(
        donnees_brutes, regroupements, verifier_modalites=True
    )
    if len(donnees_nettoyees) != len(donnees_brutes):
        raise ValueError(
            "Des individus de la CAH ont été exclus pendant la préparation "
            "(méthode de paiement Others). Vérifier le fichier source et l'appariement."
        )
    donnees_nettoyees = donnees_nettoyees.reset_index(drop=True)

    # Même algorithme que src.acm.appliquer_regroupements_experimentaux, mais
    # sans exiger qu'une modalité soit représentée dans le sous-échantillon CAH.
    # Les modalités de référence sont contrôlées ci-dessous.
    for variable, groupes in fusions.items():
        if variable not in donnees_nettoyees:
            raise KeyError(f"Variable à fusionner absente : {variable}")
        for nouvelle_modalite, anciennes_modalites in groupes.items():
            if not anciennes_modalites or len(set(anciennes_modalites)) != len(anciennes_modalites):
                raise ValueError(f"Fusion mal définie pour {variable} : {nouvelle_modalite}")
            donnees_nettoyees[variable] = donnees_nettoyees[variable].replace(
                {ancienne: nouvelle_modalite for ancienne in anciennes_modalites}
            )

    absentes = [variable for variable in variables if variable not in donnees_nettoyees]
    if absentes:
        raise ValueError(f"Variables actives de l'ACM absentes des données : {absentes}")
    donnees_acm = donnees_nettoyees.loc[:, variables].copy()
    if donnees_acm.isna().any().any():
        raise ValueError("Des modalités manquantes subsistent dans les variables de l'ACM.")

    modalites_reference = set(modalites)
    for variable in variables:
        valeurs = donnees_acm[variable].astype(str)
        observables = {f"{variable}__{valeur}" for valeur in valeurs.unique()}
        inconnues = observables - modalites_reference
        if inconnues:
            raise ValueError(
                f"Modalités incompatibles avec l'ACM 3 pour {variable} : {sorted(inconnues)}"
            )
    variables_modalites = {valeur.split("__", 1)[0] for valeur in modalites}
    if variables_modalites != set(variables):
        raise ValueError("Les modalités exportées ne correspondent pas aux dix variables actives.")
    return donnees_acm, metadonnees


def choisir_variables_descriptives(
    donnees,
    maximum_modalites=25,
):
    """Identifie les variables descriptives selon leur type."""

    variables_qualitatives = []
    variables_quantitatives = []

    for variable in donnees.columns:
        if variable == "id_anonyme":
            continue

        serie = donnees[variable]

        if is_numeric_dtype(serie):
            variables_quantitatives.append(variable)
        elif serie.nunique(dropna=True) <= maximum_modalites:
            variables_qualitatives.append(variable)

    return variables_qualitatives, variables_quantitatives


def tailles_partitions(partitions: pd.DataFrame, valeurs_k: list[int]) -> pd.DataFrame:
    """Retourne les effectifs et parts des groupes de toutes les partitions."""
    lignes = []
    for k in valeurs_k:
        colonne = f"groupe_k{k}"
        if colonne not in partitions:
            raise KeyError(f"Partition absente : {colonne}")
        effectifs = partitions[colonne].value_counts().sort_index()
        for groupe, effectif in effectifs.items():
            lignes.append({"k": k, "Groupe": int(groupe), "Effectif": int(effectif),
                           "Part (%)": 100 * int(effectif) / len(partitions)})
    return pd.DataFrame(lignes)


def profils_modalites(
    donnees: pd.DataFrame, etiquettes: np.ndarray, variables: list[str],
) -> pd.DataFrame:
    """Mesure la sur/sous-représentation de modalités dans chaque groupe."""
    etiquettes = np.asarray(etiquettes)
    if len(donnees) != len(etiquettes):
        raise ValueError("Le nombre de lignes et d'étiquettes doit être identique.")
    manquantes = set(variables).difference(donnees.columns)
    if manquantes:
        raise KeyError(f"Variables absentes : {sorted(manquantes)}")
    lignes = []
    groupes, effectifs = np.unique(etiquettes, return_counts=True)
    for variable in variables:
        valeurs = donnees[variable].astype("string").fillna("Non renseigné").reset_index(drop=True)
        frequences_generales = valeurs.value_counts(normalize=True)
        for modalite, part_generale in frequences_generales.items():
            masque = valeurs.eq(modalite).to_numpy(dtype=bool)
            for groupe, effectif in zip(groupes, effectifs):
                groupe_masque = etiquettes == groupe
                part_groupe = masque[groupe_masque].mean()
                lignes.append({"Variable": variable, "Modalité": str(modalite),
                               "Groupe": int(groupe), "Effectif groupe": int(effectif),
                               "Effectif modalité groupe": int(masque[groupe_masque].sum()),
                               "Part groupe (%)": 100 * part_groupe,
                               "Part population (%)": 100 * part_generale,
                               "Écart (points)": 100 * (part_groupe - part_generale),
                               "Indice de représentation": part_groupe / part_generale if part_generale else np.nan})
    return pd.DataFrame(lignes)


def caracteristiques_principales(
    profils: pd.DataFrame, nombre_par_groupe: int = 8, effectif_minimal: int = 10,
) -> pd.DataFrame:
    """Sélectionne les modalités les plus surreprésentées pour chaque groupe."""
    candidats = profils[profils["Effectif modalité groupe"] >= effectif_minimal].copy()
    candidats = candidats[candidats["Écart (points)"] > 0]
    return (candidats.sort_values(["Groupe", "Écart (points)"], ascending=[True, False])
            .groupby("Groupe", sort=True).head(nombre_par_groupe).reset_index(drop=True))


def profils_quantitatifs(
    donnees: pd.DataFrame, etiquettes: np.ndarray, variables: list[str],
) -> pd.DataFrame:
    """Compare les médianes et moyennes des variables numériques par groupe."""
    if len(donnees) != len(etiquettes):
        raise ValueError("Individus et étiquettes désalignés.")
    lignes = []
    for variable in variables:
        if variable not in donnees:
            raise KeyError(f"Variable absente : {variable}")
        serie = pd.to_numeric(donnees[variable], errors="coerce")
        for groupe in sorted(np.unique(etiquettes)):
            sous_serie = serie[np.asarray(etiquettes) == groupe]
            lignes.append({"Variable": variable, "Groupe": int(groupe),
                           "Moyenne groupe": sous_serie.mean(),
                           "Médiane groupe": sous_serie.median(),
                           "Moyenne population": serie.mean(),
                           "Médiane population": serie.median(),
                           "Effectif renseigné": int(sous_serie.notna().sum())})
    return pd.DataFrame(lignes)


def transitions_groupes(partitions: pd.DataFrame, k_source: int, k_cible: int) -> pd.DataFrame:
    """Compte les passages entre deux découpages de la même hiérarchie."""
    depart = f"groupe_k{k_source}"
    arrivee = f"groupe_k{k_cible}"
    if depart not in partitions or arrivee not in partitions:
        raise KeyError("Partition manquante pour la comparaison.")
    tableau = (partitions.groupby([depart, arrivee], dropna=False).size()
               .rename("Effectif").reset_index())
    effectifs_source = partitions[depart].value_counts()
    tableau["Part du groupe source (%)"] = tableau.apply(
        lambda ligne: 100 * ligne["Effectif"] / effectifs_source[ligne[depart]], axis=1)
    return tableau.sort_values([depart, "Effectif"], ascending=[True, False]).reset_index(drop=True)


def differences_enfants(
    donnees: pd.DataFrame, partitions: pd.DataFrame, k_parent: int, k_enfants: int,
    variables: list[str], nombre_modalites: int = 10,
) -> pd.DataFrame:
    """Compare les deux sous-groupes issus de chaque scission k -> k+1."""
    if k_enfants != k_parent + 1:
        raise ValueError("Les partitions doivent être consécutives.")
    transitions = transitions_groupes(partitions, k_parent, k_enfants)
    colonne_parent, colonne_enfant = f"groupe_k{k_parent}", f"groupe_k{k_enfants}"
    lignes = []
    for parent, sous_table in transitions.groupby(colonne_parent):
        enfants = sous_table[colonne_enfant].tolist()
        if len(enfants) != 2:
            continue
        indices_parent = partitions[colonne_parent].eq(parent).to_numpy()
        profils = profils_modalites(
            donnees.loc[indices_parent].reset_index(drop=True),
            partitions.loc[indices_parent, colonne_enfant].to_numpy(), variables,
        )
        pivote = profils.pivot_table(index=["Variable", "Modalité"], columns="Groupe",
                                    values="Part groupe (%)", aggfunc="first")
        enfant_a, enfant_b = sorted(enfants)
        if enfant_a not in pivote or enfant_b not in pivote:
            continue
        pivote = pivote.assign(difference_absolue=(pivote[enfant_a] - pivote[enfant_b]).abs())
        pivote = pivote.sort_values("difference_absolue", ascending=False).head(nombre_modalites)
        for (variable, modalite), ligne in pivote.iterrows():
            lignes.append({"k parent": k_parent, "Groupe parent": int(parent),
                           "k enfants": k_enfants, "Groupe A": int(enfant_a),
                           "Groupe B": int(enfant_b), "Variable": variable,
                           "Modalité": modalite,
                           "Part A (%)": ligne[enfant_a], "Part B (%)": ligne[enfant_b],
                           "Différence absolue (points)": ligne["difference_absolue"]})
    return pd.DataFrame(lignes)
