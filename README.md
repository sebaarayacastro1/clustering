# Clustering Karafun

## Objectif

Ce projet vise à segmenter les clients Karafun à partir d'une analyse des correspondances multiples (ACM), puis d'une classification ascendante hiérarchique (CAH). Il prévoit également une méthode pour affecter les clients non utilisés lors de la construction des groupes, ainsi que les nouveaux clients, sans relancer l'ensemble de la classification.

Le code est organisé en modules réutilisables et en notebooks consacrés à chaque étape de l'analyse.

## Environnement de développement

Le projet utilise **Python 3.11**, indiqué dans `.python-version`, et un **environnement virtuel local `.venv`**. Les dépendances sont définies dans `requirements.txt`. Aucun gestionnaire d'environnement supplémentaire (comme `uv`) n'est nécessaire.

Chaque collaborateur crée son propre environnement virtuel. Le dossier `.venv/` ne doit pas être versionné.

### Prérequis

- Git ;
- Python 3.11 ;
- Visual Studio Code avec les extensions **Python** et **Jupyter**, ou JupyterLab.

### 1. Cloner le dépôt

```bash
git clone git@github.com:sebaarayacastro1/clustering.git
cd clustering
```

### 2. Créer l'environnement virtuel

**Windows (PowerShell) :**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS :**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Si PowerShell bloque l'activation, il est aussi possible d'utiliser directement `.venv\Scripts\python.exe` pour installer les dépendances et d'indiquer cet interpréteur à VS Code.

### 3. Installer les dépendances

Depuis la racine du dépôt, après activation de l'environnement :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Vérifier l'environnement :

```bash
python --version
python -m pip check
```

La version affichée doit être **Python 3.11.x**.

> **Reproductibilité :** `requirements.txt` liste actuellement les bibliothèques sans versions figées. Tous les collaborateurs utilisent donc la même liste de dépendances, mais pas nécessairement les mêmes versions exactes. Si une reproductibilité stricte est nécessaire, il faudra fixer les versions après validation d'un environnement fonctionnel.

### 4. Sélectionner l'interpréteur dans VS Code

1. Ouvrir le dossier du projet dans VS Code.
2. Exécuter **Python: Select Interpreter** depuis la palette de commandes.
3. Sélectionner l'interpréteur de `.venv` (`.venv\Scripts\python.exe` sous Windows, `.venv/bin/python` sous Linux/macOS).
4. Dans chaque notebook, choisir le même environnement comme **noyau Jupyter**.

### 5. Accéder aux données et exécuter les notebooks

Le fichier **`td_clustering_karafun.csv` est déjà présent dans le dépôt Git** : il n'est pas nécessaire de le télécharger ou de le copier séparément après le clonage. Le code attend ce fichier à l'emplacement `data/brutes/td_clustering_karafun.csv` ; vérifier que le chemin utilisé dans les notebooks et la configuration correspond bien à son emplacement dans le dépôt.

Ouvrir et exécuter les notebooks dans l'ordre :

1. `notebooks/01_exploration.ipynb` : exploration des données ;
2. `notebooks/02_acm.ipynb` : itérations de l'ACM et comparaison des modalités.

Pour travailler avec JupyterLab plutôt que VS Code :

```bash
jupyter lab
```

### 6. Exécuter les tests

```bash
python -m pytest
```

## Organisation du dépôt

```text
clustering/
├── config/
│   ├── parametres.yaml
│   └── regroupements.yaml
├── data/
│   ├── brutes/
│   │   └── td_clustering_karafun.csv
│   └── traitees/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_acm.ipynb
│   ├── 03_cah.ipynb
│   ├── 04_interpretation_clusters.ipynb
│   └── 05_classification.ipynb
├── src/
│   ├── preparation.py
│   ├── acm.py
│   ├── visualisation.py
│   ├── proximite_modalites.py
│   ├── classification_hierarchique.py
│   ├── interpretation.py
│   └── affectation_clients.py
├── tests/
├── resultats/
├── modeles/
├── archives/
│   └── notebook_original_execute.ipynb
├── .python-version
├── requirements.txt
└── README.md
```

- **`config/`** : paramètres des itérations et règles de recodification.
- **`data/`** : jeu de données d'origine et données préparées.
- **`notebooks/`** : exécution, visualisation et interprétation des analyses.
- **`src/`** : fonctions Python réutilisables, indépendantes des notebooks.
- **`tests/`** : vérification des transformations et des fonctions.
- **`resultats/`** : tableaux, figures et sorties exportées.
- **`modeles/`** : objets entraînés utiles à la réutilisation de la segmentation.
- **`archives/`** : conservation du notebook historique pour comparaison.

## Méthodologie

### Étape 1 — ACM et préparation des modalités

- Sélection d'un **échantillon de 20 %**, avec une graine fixée à **42**.
- Réalisation de l'ACM à partir du tableau disjonctif complet, avec `prince.CA`.
- **Critère de sélection des variables demandé dans le cadre du cours :** une variable est conservée si au moins l'une de ses modalités présente une contribution supérieure à `100 / K` sur au moins l'un des **cinq premiers axes** (`K` : nombre total de modalités actives).
- Dans la **deuxième itération**, la variable `zone_geo_rec` a été retirée.
- Étude de la proximité des modalités d'une même variable au moyen des distances factorielles et de la distance de Jensen-Shannon. Aucun regroupement n'est appliqué automatiquement : les propositions doivent être examinées et justifiées.

### Étape 2 — CAH (à réaliser)

- Utiliser les coordonnées factorielles issues de l'ACM, après validation de la sélection des variables et des éventuels regroupements.
- Construire la CAH sur l'échantillon afin de maîtriser le coût de calcul.
- Choisir un nombre de clusters **k**, puis comparer également **k − 1** et **k + 1** du point de vue statistique et métier.
- Décrire et interpréter les groupes retenus.

Le notebook d'origine préparait quatre coordonnées individuelles pour la CAH ; le nombre d'axes définitif reste à valider.

### Étape 3 — Affectation des autres clients et des nouveaux entrants (à réaliser)

- Conserver les transformations et regroupements retenus.
- Entraîner un modèle supervisé, par exemple une **régression logistique multinomiale**, sur les segments issus de la CAH.
- Évaluer la capacité du modèle à reproduire les classes de référence.
- Appliquer le modèle aux clients restants et aux nouveaux clients sans recalculer intégralement l'ACM et la CAH.

## État actuel et points de vigilance

- Les analyses réalisées portent sur l'**ACM et la comparaison des modalités**. Les notebooks `03_cah.ipynb`, `04_interpretation_clusters.ipynb` et `05_classification.ipynb` documentent des étapes à développer ; ils ne correspondent pas encore à une chaîne de segmentation finalisée.
- La migration du notebook historique vers les modules n'a pas encore été validée numériquement sur le CSV dans l'environnement de préparation : les résultats doivent être comparés après exécution locale.
- Le calcul historique de l'inertie corrigée de **Benzécri** repose sur les composantes calculées (15) : les pourcentages sont à vérifier avec l'ensemble des valeurs propres utiles.
- Les sorties du notebook historique sont conservées pour faciliter les comparaisons.

## Bonnes pratiques de collaboration

- Versionner `README.md`, `.python-version`, `requirements.txt`, les fichiers de configuration, les modules et les notebooks.
- **Ne pas versionner `.venv/`**, les caches Python et les fichiers temporaires de Jupyter.
- Le CSV d'origine étant déjà suivi dans ce dépôt, éviter de créer des copies inutiles des données ou d'ajouter de nouveaux exports volumineux sans concertation.