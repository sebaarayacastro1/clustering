# Clustering Karafun

## Objectif

Ce projet vise à **segmenter les clients Karafun** selon leurs caractéristiques et leurs comportements, à partir d'une **analyse des correspondances multiples (ACM)** suivie d'une **classification ascendante hiérarchique (CAH)** avec la méthode de Ward.

La segmentation retenue comprend **cinq groupes métier**. La suite du projet consiste à entraîner un classifieur supervisé pour affecter les autres clients, puis à proposer une interface **Streamlit** permettant de classer un nouveau client ou de réévaluer un client existant **sans recalculer la CAH ni réajuster l'ACM**.

Le code est organisé en modules réutilisables (`src/`) et en notebooks correspondant aux différentes étapes.

## État d'avancement

| Étape | Statut | Résultat ou objectif |
| --- | --- | --- |
| **1. Choix et interprétation des clusters** | **Terminée pour le périmètre retenu** | ACM définitive, CAH sur 20 000 clients, choix métier de **k = 5**. |
| **2. Entraînement d'un classifieur** | À réaliser | Apprendre à reproduire les étiquettes de la CAH et évaluer les prédictions. |
| **3. Affectation du reste de la base** | À réaliser | Prédire les groupes des clients non inclus dans la CAH de référence. |
| **4. Outil réutilisable** | À réaliser | Interface Streamlit de prédiction et de mise à jour des clients. |

> **Périmètre de la décision :** la partition de référence repose sur une CAH de **20 000 clients**, les **quatre premiers axes** de l'ACM définitive, la méthode **Ward**, **k = 5** et une **graine aléatoire fixée à 42**. La décision privilégie l'interprétabilité des groupes et leur intérêt métier ; elle ne constitue pas une preuve d'optimalité ou de stabilité universelle.

## Environnement de développement

Le projet utilise **Python 3.11**, indiqué dans `.python-version`, et un environnement virtuel local **`.venv`**. Les dépendances figurent dans `requirements.txt`. Aucun gestionnaire d'environnement supplémentaire, tel que `uv`, n'est nécessaire.

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

Si PowerShell bloque l'activation, il est aussi possible d'utiliser directement `.venv\Scripts\python.exe` et de sélectionner cet interpréteur dans VS Code.

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Vérifier l'environnement :

```bash
python --version
python -m pip check
```

La version de Python doit être **3.11.x**.

> **Reproductibilité :** si `requirements.txt` ne fixe pas les versions exactes, deux collaborateurs peuvent installer des versions différentes des bibliothèques. Il conviendra de figer les versions validées avant la livraison opérationnelle.

### 4. Sélectionner l'interpréteur dans VS Code

1. Ouvrir le dossier du projet dans VS Code.
2. Exécuter **Python: Select Interpreter** depuis la palette de commandes.
3. Sélectionner l'interpréteur de `.venv` (`.venv\Scripts\python.exe` sous Windows ou `.venv/bin/python` sous Linux/macOS).
4. Choisir ce même environnement comme **noyau Jupyter** dans les notebooks.

### 5. Données et ordre des notebooks

Le jeu de données d'origine est disponible à l'emplacement `data/brutes/td_clustering_karafun.csv` dans l'organisation actuelle du dépôt. Vérifier la présence de ce fichier et les chemins utilisés par les notebooks.

| Notebook | Rôle | État |
| --- | --- | --- |
| `01_exploration.ipynb` | Exploration des données clients | Réalisé |
| `02_acm.ipynb` | Itérations et choix de l'ACM définitive | Réalisé |
| `03_cah.ipynb` | CAH avec Ward et comparaison des partitions | Réalisé |
| `04_interpretation_clusters.ipynb` | Profils métier, scissions et choix de **k = 5** | Réalisé |
| `05_classification.ipynb` | Entraînement du classifieur supervisé | **Prochaine étape : à développer** |

Pour utiliser JupyterLab :

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
│   └── cah/
├── modeles/
├── archives/
│   └── notebook_original_execute.ipynb
├── .python-version
├── requirements.txt
└── README.md
```

- **`config/`** : paramètres de calcul et règles de recodification des modalités.
- **`data/`** : données brutes et préparées.
- **`notebooks/`** : expérimentations, visualisations, interprétation et modélisation.
- **`src/`** : fonctions réutilisables, indépendantes des notebooks.
- **`tests/`** : vérification des transformations et des fonctions.
- **`resultats/`** : tableaux, figures et partitions exportées.
- **`modeles/`** : modèles et métadonnées nécessaires à la réutilisation.
- **`archives/`** : conservation de travaux historiques.

Les futurs modules de prédiction et l'application Streamlit seront ajoutés lors des étapes correspondantes ; ils ne font pas encore partie de la chaîne opérationnelle.

## Méthodologie et résultats

### 1. Analyse des correspondances multiples (ACM)

Une première analyse a été menée sur un échantillon représentant environ **20 % des données clients disponibles**, soit **68 218 individus** dans le jeu analysé. Plusieurs itérations ont permis d'étudier les contributions des modalités et de sélectionner les variables actives.

Le critère de contribution étudié dans le cadre du cours consiste à vérifier si au moins une modalité d'une variable dépasse le seuil **`100 / K`** sur au moins l'un des cinq premiers axes (`K` désignant le nombre de modalités actives). La variable `zone_geo_rec` a été écartée lors de la deuxième itération.

L'étude des proximités factorielles de modalités appartenant à une même variable a conduit à retenir **un seul regroupement supplémentaire** pour l'ACM définitive : la fusion des catégories **« 10–20 »** et **« 20 et plus »** de `avg_songs_sessions_rec`. Les autres regroupements envisagés n'ont pas été appliqués.

L'**ACM définitive (troisième itération)** repose sur **10 variables actives et 42 modalités**. Les coordonnées factorielles et le modèle ACM ont été exportés afin d'être réutilisés pour la CAH et, ultérieurement, pour la projection de nouveaux clients.

### 2. Classification ascendante hiérarchique (CAH)

La classification a été réalisée par **la méthode de Ward et la distance euclidienne**, sur les **quatre premiers axes factoriels** issus de l'ACM définitive. Pour maîtriser le coût de calcul, une **sous-population de 20 000 clients**, sélectionnée avec une **graine fixée à 42**, sert de référence.

Les partitions en **4, 5 et 6 groupes** ont été comparées à l'aide du dendrogramme, des effectifs, de coefficients de silhouette exploratoires et de l'analyse des scissions successives.

| Nombre de groupes | Silhouette exploratoire |
| --- | ---: |
| **4** | 0,350 |
| **5** | 0,336 |
| **6** | 0,354 |

Ces valeurs proviennent de l'expérience de 20 000 clients et sont utilisées comme **indicateurs complémentaires**, sans déterminer automatiquement le choix métier.

### 3. Interprétation des groupes et choix de k

Les profils ont été étudiés **à partir des mêmes dix variables actives et des mêmes recodifications que l'ACM définitive**. Les fréquences des modalités par groupe ont été comparées aux fréquences globales ; les écarts sont exprimés en **points de pourcentage**.

Le passage de **4 à 5 groupes** permet de distinguer deux profils auparavant réunis : des clients avec une faible fidélité et une faible valeur, et des clients présentant davantage de comportements intermédiaires et de réactivations.

Le passage de **5 à 6 groupes** sépare principalement les tout nouveaux abonnés des autres clients récents. Cette distinction peut être pertinente pour les actions d'accueil, mais l'équipe a choisi de **conserver un segment commun pour les clients récents** et de traiter, si nécessaire, l'onboarding par des actions spécifiques.

**Décision retenue : `k = 5`**, pour la lisibilité et l'utilité métier de la segmentation, malgré une silhouette légèrement inférieure à celles de `k = 4` et `k = 6`.

#### Profils métier retenus

| Groupe | Nom métier proposé | Effectif de référence |
| --- | --- | ---: |
| **G1** | Clients fidèles à forte valeur | 5 134 |
| **G2** | Clients en churn à faible fidélité | 4 152 |
| **G3** | Clients sans activité observée | 2 201 |
| **G4** | Clients intermédiaires souvent réactivés | 4 060 |
| **G5** | Clients récents en phase d'intégration | 4 453 |
| **Total** | | **20 000** |

Les noms décrivent les tendances dominantes des groupes : ils ne signifient pas que tous leurs membres possèdent chacune des caractéristiques indiquées. La numérotation G1–G5 est liée à **cette partition de référence** ; elle ne doit pas être transférée telle quelle à une autre exécution de la CAH.

### 4. Limites de l'analyse

- La CAH de référence utilise **20 000 clients**, et non l'ensemble des 68 218 individus de l'ACM.
- Les silhouettes sont **exploratoires** et ne suffisent pas à établir un nombre de groupes optimal.
- Le résultat dépend du choix des axes factoriels et peut être sensible à l'échantillonnage. **Aucune forte stabilité de la partition n'est revendiquée** dans le périmètre retenu.
- L'étiquetage métier décrit des profils observés ; il ne démontre pas l'efficacité des actions commerciales envisagées.

## Étapes suivantes

### Étape 2 — Entraîner un classifieur

Le prochain travail sera réalisé dans **`notebooks/05_classification.ipynb`**, qui reste **à développer**. La partition CAH de référence (`k = 5`, graine 42) servira de source d'étiquettes pour les 20 000 clients.

Le classifieur utilisera les **quatre coordonnées factorielles** correspondantes comme variables explicatives. Une **régression logistique multinomiale** constituera un premier modèle de référence, éventuellement comparé à une méthode non linéaire.

Une évaluation sur des observations mises de côté mesurera notamment l'exactitude, le **F1-score macro**, le rappel par groupe et les confusions entre segments. Cette évaluation mesurera la capacité à **reproduire les étiquettes de la CAH**, et non la validité intrinsèque de la segmentation.

### Étape 3 — Affecter le reste de la base

Après validation du classifieur, les **48 218 clients de l'échantillon ACM non inclus dans les 20 000 clients de référence** pourront recevoir un groupe prédit. Les sorties distingueront les **étiquettes CAH de référence** des **affectations prédites**.

### Étape 4 — Proposer une application Streamlit

Une application Streamlit devra permettre de saisir les caractéristiques d'un nouveau client ou de mettre à jour celles d'un client existant, puis de prédire son groupe.

Le traitement réutilisera les **mêmes règles de préparation et de recodification**, l'**ACM ajustée** pour projeter les observations sur les quatre axes et le **classifieur enregistré**. Ni la CAH ni l'ACM ne devront être réajustées à chaque prédiction. La gestion des modalités inconnues, des historiques insuffisants et de l'enregistrement des mises à jour devra être définie avant la mise en service.

Une **API séparée n'est pas indispensable** : Streamlit pourra appeler directement les fonctions de prédiction du projet.

## Bonnes pratiques de collaboration

- Versionner `README.md`, `.python-version`, `requirements.txt`, les fichiers de configuration, les modules et les notebooks.
- **Ne pas versionner `.venv/`**, les caches Python ni les fichiers temporaires Jupyter.
- Éviter les copies inutiles du CSV d'origine et les exports volumineux non concertés.
- Conserver les paramètres de la **partition de référence** et vérifier la correspondance entre les identifiants des clients et leurs coordonnées factorielles.
- Ne pas modifier les règles de l'ACM définitive entre l'entraînement du classifieur et la prédiction de nouveaux clients.
