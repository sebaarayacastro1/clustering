"""Préparation de la classification ascendante hiérarchique (étape suivante)."""

def extraire_coordonnees(resultats_acm, nombre_axes=4):
    """Retourne les coordonnées des individus, selon le notebook historique."""
    modele = resultats_acm['modele']
    tableau_disjonctif = resultats_acm['tableau_disjonctif']
    return modele.row_coordinates(tableau_disjonctif).iloc[:, :nombre_axes]

# La CAH et le choix de k ne sont pas implémentés dans le notebook fourni.
# Aucune partition n'est créée implicitement ici.
