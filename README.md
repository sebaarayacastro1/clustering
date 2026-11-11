# Clustering Karafun

## Objectif

Ce projet vise à **segmenter les clients Karafun** selon leurs caractéristiques et leurs comportements, à partir d'une **analyse des correspondances multiples (ACM)** suivie d'une **classification ascendante hiérarchique (CAH)** avec la méthode de Ward.

La segmentation retenue comprend **cinq groupes métier**. Une **régression logistique multinomiale** reproduit les groupes CAH sur les quatre coordonnées de l'ACM. Les affectations ont été étendues à la base historique. Une première interface **Streamlit** permet désormais de prédire le segment d'un client à partir des modalités finales de l'ACM, sans recalculer la CAH ni réajuster les modèles. L'enregistrement et la modification persistante des clients ne font pas partie du périmètre initial ; ils peuvent faire l'objet d'une évolution ultérieure.

Le code est organisé en modules réutilisables (`src/`) et en notebooks correspondant aux différentes étapes.

## État d'avancement

| Étape | Statut | Résultat |
| --- | --- | --- |
| **1. Choix et interprétation des clusters** | **Terminée** | ACM 3 : 10 variables actives et 42 modalités ; CAH Ward sur 20 000 clients ; **k = 5**. |
| **2. Classification supervisée** | **Terminée** | Régression logistique multinomiale ; accuracy de **95,05 %** et F1 macro de **95,33 %** sur 4 000 clients de test. |
| **3. Extrapolation à la base complète** | **Terminée** | **341 091** clients affectés sur **341 092** enregistrements ; un enregistrement exclu selon la préparation initiale. |
| **4. Application Streamlit** | **Terminée** | Interface de prédiction individuelle opérationnelle ; gestion persistante des clients facultative et hors périmètre initial. |

> **Périmètre de la décision :** la partition de référence repose sur une CAH de **20 000 clients**, les **quatre premiers axes** de l'ACM définitive, la méthode **Ward**, **k = 5** et une **graine aléatoire fixée à 42**. La décision privilégie l'interprétabilité des groupes et leur intérêt métier ; elle ne constitue pas une preuve d'optimalité ou de stabilité universelle.

## Environnement de développement

Le projet utilise **Python 3.11**, indiqué dans `.python-version`, et un environnement virtuel local **`.venv`**. Les dépendances scientifiques figurent dans `requirements.txt` ; les dépendances de l’interface sont complétées dans `app/requirements_streamlit.txt`. Aucun gestionnaire d'environnement supplémentaire, tel que `uv`, n'est nécessaire.

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

> **Reproductibilité :** `requirements.txt` fixe les versions de l’environnement scientifique de référence. La contrainte de version de Streamlit figure séparément dans `app/requirements_streamlit.txt` ; vérifier la compatibilité de l’ensemble dans l’environnement de livraison.

### 4. Sélectionner l'interpréteur dans VS Code

1. Ouvrir le dossier du projet dans VS Code.
2. Exécuter **Python: Select Interpreter** depuis la palette de commandes.
3. Sélectionner l'interpréteur de `.venv` (`.venv\Scripts\python.exe` sous Windows ou `.venv/bin/python` sous Linux/macOS).
4. Choisir ce même environnement comme **noyau Jupyter** dans les notebooks.

### 5. Données et ordre des notebooks

Le fichier source historique est attendu à l'emplacement `data/brutes/td_clustering_karafun.csv`. Les modèles et résultats exportés doivent être présents aux chemins attendus par les modules. Les fichiers volumineux ne sont pas nécessairement versionnés dans Git.

| Notebook | Rôle | État |
| --- | --- | --- |
| `01_exploration.ipynb` | Exploration des données clients | Réalisé |
| `02_acm.ipynb` | Construction et choix de l'ACM définitive | Réalisé |
| `03_cah.ipynb` | CAH Ward et comparaison des partitions | Réalisé |
| `04_interpretation_clusters.ipynb` | Profils métier, variable géographique illustrative et choix de **k = 5** | Réalisé |
| `05_classification.ipynb` | Entraînement et évaluation de la régression logistique multinomiale | Réalisé |
| `06_extrapolation_clusters.ipynb` | Affectation des clients de l'ACM et de la base complète | Réalisé |

Pour utiliser JupyterLab :

```bash
jupyter lab
```

### 6. Exécuter les tests

```bash
python -m pytest
```

Pour contrôler spécifiquement le moteur de prédiction et l'interface :

```bash
python -m pytest tests/test_prediction_client.py tests/test_prediction_client_integration.py tests/test_modalites_interface.py -v -rs
```

Les tests d'intégration s'appuient sur les exports et modèles de l'étape 3 ; ils doivent être exécutés dans l'environnement du projet contenant ces artefacts.

### 7. Lancer l'application Streamlit

Installer les dépendances complémentaires (depuis la racine du dépôt) :

```bash
python -m pip install -r app/requirements_streamlit.txt
```

Démarrer l'application :

```bash
python -m streamlit run app/streamlit_app.py
```

L'interface présente **les modalités finales de l'ACM 3** sous forme de questions et de libellés français lisibles, sans afficher les préfixes de tri techniques. Par exemple, les valeurs sources `Apple` et `Android` sont représentées par l'unique catégorie finale « Paiement sur téléphone (Apple / Android) ». Pour l'ancienneté d'abonnement, la catégorie finale `05_Plus_de_2_ans` est affichée comme « Plus de 2 ans ». Le texte affiché est distinct de la valeur source représentative envoyée au moteur, qui applique les regroupements pour restituer la modalité exacte attendue par l'ACM.

L'interface permet de **prédire le segment d'un client** et d'afficher un tableau des cinq scores estimés. **Aucune création, recherche ou modification de client dans une base de données n'est nécessaire** pour cette fonctionnalité : le formulaire n'enregistre pas les données saisies dans une base persistante. Les scores ne sont pas calibrés et ne correspondent pas à des probabilités de churn futur.

## Organisation du dépôt

```text
clustering/
├── app/
│   ├── streamlit_app.py
│   ├── modalites_interface.py
│   └── requirements_streamlit.txt
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
│   ├── 05_classification.ipynb
│   └── 06_extrapolation_clusters.ipynb
├── src/
│   ├── preparation.py
│   ├── acm.py
│   ├── classification_hierarchique.py
│   ├── classification_supervisee.py
│   ├── extrapolation_clusters.py
│   ├── prediction_client.py
│   ├── interpretation.py
│   └── ...
├── tests/
│   ├── test_prediction_client.py
│   ├── test_prediction_client_integration.py
│   ├── test_modalites_interface.py
│   └── ...
├── resultats/
│   ├── cah/
│   └── extrapolation/
├── modeles/
├── .python-version
├── requirements.txt
└── README.md
```

- **`config/`** : paramètres et règles de recodification.
- **`data/`** : données brutes et préparées.
- **`notebooks/`** : explorations, modélisation, interprétation et extrapolation.
- **`src/`** : fonctions de préparation, de projection et de prédiction réutilisables.
- **`app/`** : interface de prédiction Streamlit et adaptation des libellés.
- **`tests/`** : tests unitaires et d'intégration.
- **`resultats/`** : tableaux, figures et affectations exportées.
- **`modeles/`** : modèles sauvegardés et métadonnées nécessaires à la prédiction.

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

## Classification supervisée et extrapolation

### Étape 2 — Classifieur opérationnel

Le notebook `05_classification.ipynb` entraîne une **régression logistique multinomiale** sur les **quatre premières coordonnées factorielles** des 20 000 clients étiquetés par la CAH (G1 à G5). Sur le jeu de test de 4 000 clients, l'accuracy atteint **95,05 %** et le F1-score macro **95,33 %**.

Ces métriques mesurent la capacité du classifieur à **reproduire la partition CAH de référence**, pas une prédiction de churn futur ni une validation indépendante de la stabilité des groupes.

### Étape 3 — Extrapolation à la base historique

Le notebook `06_extrapolation_clusters.ipynb` conserve les 20 000 étiquettes CAH, classe 48 218 clients ayant déjà des coordonnées ACM et projette 272 873 autres clients avec l'ACM sauvegardée avant leur affectation supervisée.

| Origine de l'affectation | Effectif |
| --- | ---: |
| Étiquettes CAH conservées | 20 000 |
| Prédictions sur coordonnées ACM existantes | 48 218 |
| Projection ACM puis prédiction | 272 873 |
| Enregistrement exclu par la préparation (`methode_paiement == "Others"`) | 1 |
| **Total des enregistrements** | **341 092** |

**341 091 clients sont affectés** à l'un des cinq groupes. La reprojection de 200 individus connus a reproduit leurs coordonnées ACM avec un écart maximal de **5,551 × 10⁻¹⁷**. La proximité des distributions de groupes entre la CAH et la base complète indique une cohérence descriptive, sans garantir l'exactitude individuelle de chaque affectation prédite.

### Étape 4 — Application Streamlit

L'application Streamlit constitue une interface de **prédiction individuelle** : à partir des caractéristiques d'un client, elle applique les regroupements de l'ACM définitive, projette l'observation à l'aide du modèle ACM sauvegardé, puis utilise la régression logistique multinomiale pour attribuer l'un des cinq segments **G1 à G5**.

Le formulaire présente directement les **modalités finales regroupées**, avec des questions en français et des libellés lisibles. Par exemple, une catégorie d'ancienneté s'affiche sous la forme « Plus de 2 ans » plutôt que sous son code technique. La conversion interne conserve les codes attendus par les modèles sauvegardés.

Le résultat affiche le **groupe prédit, son nom métier et un tableau des cinq scores**. L'application ne recalcule ni la CAH ni l'ACM et ne réentraîne pas le classifieur. Les scores indiquent la répartition estimée entre les cinq groupes ; ils ne représentent pas un risque de churn et leur calibration n'a pas été évaluée.

Cette version répond à l'objectif de **classer un nouveau client sans refaire la segmentation**. Elle ne conserve pas les données du formulaire dans une base persistante. La gestion d'un historique de clients constitue une fonctionnalité distincte et facultative.

### Améliorations possibles — hors périmètre initial

Deux évolutions sont identifiées, sans remettre en cause le fonctionnement actuel de la classification :

1. **Préciser l'unité de la valeur client (`valeur_12M`).** La variable `valeur_12M_rec` est présentée sous forme d'intervalles numériques, mais la devise ou l'unité métier exacte n'a pas été confirmée à partir de la documentation source. Vérifier cette définition avant d'ajouter un symbole monétaire (par exemple `€`) ou de reformuler la question. Cette amélioration concerne les libellés et ne doit pas modifier les modalités transmises à l'ACM.

2. **Ajouter une gestion persistante des clients (facultatif).** Dans une évolution ultérieure, l'application pourrait permettre d'enregistrer de nouveaux clients, de rechercher les clients historiques et de modifier leurs caractéristiques en recalculant leur segment. Un stockage local **SQLite** pourrait suffire pour un prototype ; une base externe pourrait être envisagée si les besoins de déploiement évoluent. Aucune base de données ni API n'est nécessaire pour la prédiction individuelle actuelle.

## Bonnes pratiques de collaboration

- Versionner `README.md`, `.python-version`, `requirements.txt`, les fichiers de configuration, les modules et les notebooks.
- **Ne pas versionner `.venv/`**, les caches Python ni les fichiers temporaires Jupyter.
- Éviter les copies inutiles du CSV d'origine et les exports volumineux non concertés.
- Conserver les paramètres de la **partition de référence** et vérifier la correspondance entre les identifiants des clients et leurs coordonnées factorielles.
- Ne pas modifier les règles de l'ACM définitive entre l'entraînement du classifieur et la prédiction de nouveaux clients.
