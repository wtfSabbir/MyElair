# ECLAIR: Attention Fully Convolutional Semantic Segmentation Algorithm

## Sommaire

- [Installation](#installation---procédure-suivie-pour-recréer-lenvironnement)
- [Prédiction des PLOs](#prediction-des-plos-avec-eclair)
  - [Prédire des nuages de points](#prédiction-dobjets-verticaux-avec-eclair-sur-nuages-photogrammétriques-lasz)
  - [Reproduire les modèles](#reproductibilité-des-entrainements)
  - [Comparaison des résultats avec le modèle existant](#evaluation-des-résultats)
- [Prédiction de nuage LiDAR](#prédiction-de-nuage-lidar-avec-eclair)
- [Lancer un entraînement avec ECLAIR](#lancer-un-entrainement-avec-eclair)
  - [Nuages de points](#nuages-de-points)
  - [Configuration de l'entraînement](#configuration-de-lentrainement)
  - [Transformations appliquées aux nuages](#transformations-appliquées-aux-nuages)
  - [Entrainer le modèle](#entrainer-le-modèle)
- [Prédire des nuages de points](#prédire-des-nuages-de-points)
- [Intégration dans PSANP](#intégration-dans-psanp)
- [En savoir plus](#en-savoir-plus)

---

## Installation - Procédure suivie pour recréer l'environnement


Eclair utilise la bibliothèque MinkowskiEngine uniquement disponible sous Linux.

**Prérequis :**
* Distribution Linux basée sur Debian (par exemple Ubuntu)
* [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* OpenBLAS (`sudo apt install -y libopenblas-dev`)
* Python 3.10 (environnement virtuel fortement conseillé)

```bash
pip install --index-url="https://pypi.geo-sat.com/simple" eclair
```


<br>
<br>

## Prediction des PLOs avec Eclair

### Prédiction d'objets verticaux avec Eclair sur nuages photogrammétriques (`.la[s/z]`).

La prédiction se lance avec la commande :

```bash
python -m eclair.plos.predict.predict --weight_path1 [path to the basemodel weights] --weight_path2 [path to the PLOs model weights] --config_file [path to the config.yaml file] --pointclouds [path to the pointclouds folder] --save_path [path to the save folder]
```
* Les poids des modèles **--weight_path1** et **--weight_path2** sont disponibles sur MLFlow: https://gsvision.geo-sat.com/mlflow/#/experiments/911728019003067248/runs/70d9b7c1332b47af81d23a5bb8495e28/artifacts
* Le fichier **--config_file** est le fichier `src/eclair/plos/predict/config.yaml`
* Les dossiers de nuages à prédire et de sauvegarde des prédictions sont à renseigner dans les arguments 
```--config_file``` et ```--save_path```

Chaque nuage prédit et sauvegardé contient 3 nouveaux champs :

* **basemodel_v13** : contient les prédictions du modèle Eclair de base, avec les classes:
  ```text
  Unclassified: 0
  Sol: 1
  PLO: 2
  Tronc: 3
  Vegetation: 4
  Batiment: 5
  ```
* **plo_v13** : contient les prédictions du modèle Eclair spécialisé dans les PLOs, avec les classes :
  ```text
  Unclassified: 0
  Poteau: 1
  Panneau: 2
  Potelet: 3
  ```
* **fuze** : contient la fusion des prédictions précédentes, incluant des règles de décision en fonction des classes
  prédites. Le mapping des classes est le suivant :
  ```text
  Unclassified: 0
  Sol: 1
  Tronc: 2
  Vegetation: 3
  Batiment: 4
  Poteau: 5
  Panneau: 6
  Potelet: 7
  ```

> ⚠️ Le champ 'classification' est également édité avec la même classification que le champ 'fuze', afin de pouvoir le visualiser sur Potree ou Paintcloud

> ⚠️ Les nuages doivent être de taille 40x40x40 mètres maximum ! Les points au-delà seront tous classés en 'Unclassified'

---

### Reproductibilité des entrainements

Les scripts présents dans le dossier `/src/eclair/plos/train` permettent de reproduire
les entrainements des 2 modèles utilisés pour la prédiction des PLOs :
- Fichiers de configuration *first_model.yaml* pour le modèle de base et *plo.yaml* pour le modèle PLO
- Fichier *train.py* pour exécuter l'entrainement.
- Les jeux de données utilisés pour ces entrainements ainsi que les fichiers de répartition train/val
sont disponibles ici: `R:\9_temporary_files\for_temporary_storage_1_month\malo\Eclair_preprod\v13\train_data`

<br>

### Evaluation des résultats

Le script `src/eclair/plos/predict/evaluate_mlflow.py` permet d'évaluer les résultats de prédictions 
avec des nuages "vérités" annotés manuellement. Les métriques utilisées sont les scores de précision et rappel
pour les classes *Poteaux*, *Panneau*, *Potelet* et *Tronc*. Les métriques sont calculées en comparant
les champs *Classification* de deux dossiers de nuages de points `.la[s/z]`. Les nuages doivent avoir exactement le même
nom de fichier dans les deux dossiers. 

Le script s'exécute avec la commande :

```bash
python -m eclair.plos.predict.evaluate_mlflow --gt_dir [path to the ground truth pointclouds folder] --pred_dir [path to the predicted pointclouds folder] --config_path [path to the evaluate_mlflow.json file] [--log_mlflow]
```

* Le fichier de configuration est disponible ici: `src/eclair/plos/predict/evaluate_mlflow.json`
  * Le paramètre **log_artifacts_files** de ce fichier permet de stoker des fichiers
  liés à cette expérience (poids des modèles ou fichier de configuration des entrainements par exemple).
  Il attend une liste de chemins vers les fichiers que vous souhaitez sauvegarder. Par défaut, aucun fichier n'est sauvegardé.
* Les nuages 'vérités' annotés manuellement sont stockés ici: `R:\9_temporary_files\for_temporary_storage_1_month\malo\Eclair_preprod\v13\pointclouds_test_gt`
* L'argument ```--log_mlflow``` active l'enregistrement des résultats sur MLflow

Les différentes expérimentations sur la segmentation des PLOs sont disponibles ici : https://gsvision.geo-sat.com/mlflow/#/experiments/911728019003067248



<br>
<br>


### Prédiction de nuage LiDAR avec Eclair

Un modèle Eclair est disponible pour la segmentation de nuage LiDAR. Les classes segmentées sont les suivantes : 

  ```text
  Unclassified: 0
  Sol: 1
  Végétation: 2
  Batiment: 3
  Humain: 4
  Mobilier urbain: 5
  Véhicule: 6
  Cable: 7
  ```

Les poids, scripts d'entrainement et prédiction, ainsi que l'emplacement du jeu d'entrainement sont disponibles sur MLFlow :
https://gsvision.geo-sat.com/mlflow/#/experiments/433491281747178879/runs/85a12cadffb8468491bf86f71be0bab0

> ⚠️ Pour utiliser ce modèle, les nuages doivent être de taille 30x30x30 mètres maximum ! Les points au dela seront tous classés en 'Unclassified'


<br>
<br>

## Lancer un entrainement avec ECLAIR

Les scripts présents dans le dossier /scripts permettent de lancer un entrainement d'Eclair et de prédire des nuages de points
avec un modèle entrainé. 

### Nuages de points

Les nuages de points doivent être au format .la[s/z]
* Un fichier .json avec les noms de vos nuages et leurs répartitions en train/val/
  test est requis. Le fichier doit avoir la forme :

```json
[
  {
    "tile_name": "pointcloud1.laz",
    "split": "train"
  },
  {
    "tile_name": "pointcloud2.laz",
    "split": "test"
  },
  {
    "tile_name": "pointcloud3.laz",
    "split": "val"
  },
  ...
]
```
* Les nuages en train et val doivent être placés dans le même dossier qui sera à renseigner pour l'entrainement.

<br>

### Configuration de l'entrainement

Toute la configuration de l'entrainement s'effectue avec le fichier scripts/config.yaml


#### Paramètres liés aux données

- Le paramètre **features_name** liste les features à utiliser pour entrainer le modèle. Il suffit de commenter ceux non désirés dans le fichier de config. Quatre features sont pris en charge :
    - L'intensité (intensity)
    - Les valeurs RGB (colors)
    - Le return number (return_number)
    - Le number of returns (number_of_returns)
  
  

- Le paramètre **num_features** doit contenir le nombre de features total pour votre modèle. Exemple : si vous voulez
  prendre en compte l'intensité et les valeurs RGB en features, le nombre total de features est de 4 (intensité=1,
  RGB=3)

```yaml
feature_names:
  - intensity # size: 1
  #  - return_number # size: 5 (one-hot)
  #  - number_of_returns # size: 5 (one-hot)
  - colors # size: 3

num_features: 4
```

<br>

- Le paramètre "num_classes" indique le nombre de classes pour votre modèle. Attention, si vous n'avez que 2
  classes (une classe vs tout le reste), il faut indiquer num_classes = 1

```yaml
num_classes: 3  # Si deux classes, indiquer 1. Sinon, indiquer normalement le nb de classes.
```

<br>

* Si vous souhaitez réordonner vos classes, en fusionner ou ne réaliser l'entrainement qu'avec certaines, vous
  pouvez utiliser le paramètre **input_mapping**. Il permet de réaliser un mapping entre les ids des classes de vos nuages
  et ceux qui seront réellement utilisés lors de l'entrainement. Les ids à la sortie d'**input_mapping** doivent commencer à zeros et être consécutifs.

```yaml
#Exemple pour passer de 11 à 3 classes

input_mapping:
  0: 0
  1: 1
  2: 0
  3: 1
  4: 0
  5: 2
  6: 2
  7: 2
  8: 0
  9: 0
  10: 0
```

* Si vous ne souhaitez pas toucher aux classes de vos nuages, il suffit de mapper chaque id vers lui-même

```yaml
input_mapping:
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4
  5: 5
  6: 6
  7: 7
  8: 8
  9: 9
  10: 10
```

<br>

#### Paramètres du modèle

* Le paramètre **model_name** définit le modèle utilisé. Douze modèles sont disponibles. Tous ces modèles sont basés sur l’architecture U-Net pour les données 3D, en utilisant la
librairie MinkowskiEngine.
Ils partagent une structure en encoder/decoder, mais diffèrent par leur profondeur (nombre de couches) et leur largeur
(nombre de canaux), ce qui influence directement leur capacité d’apprentissage, temps d'entraînement et performance.
Les variantes d’un même modèle (comme 18A, 18B, 18D) ajustent les canaux pour s’adapter à différentes contraintes de
ressources ou besoins en précision. Voici les modèles disponibles ainsi qu'une brève description de chacun :

| Modèle      | Description                                                                                                   |
|-------------|---------------------------------------------------------------------------------------------------------------|
| MinkUNet14  | Modèle très léger avec peu de blocs, idéal pour des tests ou de petits jeux de données.                       |
| MinkUNet14C | Variante du 14 avec une largeur augmentée (PLANES personnalisés) pour un meilleur compromis capacité/vitesse. |
| MinkUNet18  | Modèle standard équilibré, couramment utilisé pour un bon compromis performance/complexité.                   |
| MinkUNet18A | Variante allégée du 18 avec moins de canaux dans le décodeur, pour une exécution plus rapide.                 |
| MinkUNet18B | Variante plus homogène du 18 avec des canaux constants dans le décodeur.                                      |
| MinkUNet18D | Variante plus large du 18, avec beaucoup plus de canaux en sortie (décodeur).                                 |
| MinkUNet34  | Version plus profonde que le 18, adaptée à des tâches plus complexes.                                         |
| MinkUNet34A | Variante du 34 avec un décodeur plus large, conservant une symétrie d’architecture.                           |
| MinkUNet34B | Variante du 34 avec une réduction progressive des canaux dans le décodeur.                                    |
| MinkUNet34C | Variante du 34 avec un décodeur modérément réduit, entre A et B.                                              |
| MinkUNet50  | Modèle profond utilisant des blocs bottleneck, plus efficace en mémoire pour sa taille.                       |
| MinkUNet101 | Très grand modèle à blocs bottleneck, adapté à des tâches nécessitant une forte capacité de représentation.   |

* **voxel_size**: taille des voxels, en mètre.
* **batch_size**: taille des batchs en entrainement.
* **test_batch_size**: taille des batchs en test.
* **epoch**: nombre d'epochs pour l'entrainement 
* **patience**: nombre d'epochs consécutives sans amélioration de la validation loss avant que l'entrainement s'arrête.
* **num_worker**: nombre de processus en parallèle pour lire les données.
* **pin_memory**: active l'optimisation pour un transfert plus rapide des données vers le GPU (True/False)
* **learning_rate**: le 'pas' utilisé pour mettre à jour les poids du modèle à chaque itération.
* **weight_decay**: pénalité appliquée à la taille des poids du modèle (réduit l'overfitting).
* **self_attention**: insère 0, 1 ou 2 modules de self attention au modèle.
* **multihead_attention**: insère 0, 1 ou 2 modules de multihead attention au modèle.
* **num_head**: nombre de tête des modules de multihead attention.
* **loss**: loss souhaitée pour l'entrainement. Trois loss sont disponibles:
    * Pour une classification binaire (id 0 : tout ce qui n'est pas recherché, id 1 : la classe recherchée), il est conseillé d'utiliser la *TverskyFocaleLoss*.
    * Pour un nombre de classes >2, la *FocaleLoss* et la *FocalTverskyLoss_multiclass* sont proposées.
* **loss_weights**: Les hyper-paramètres de la loss.
    * Pour la *TverskyFocaleLoss*, **loss_weights** doit être de la forme [alpha, beta]. Pour votre classe avec l'id 1, alpha
      pénalisera les faux positifs et beta les faux négatifs (0 < alpha < 1 & 0 < beta < 1).
    * Pour la *FocalTverskyLoss_multiclass*, **loss_weights** doit avoir la forme: [[alpha_0, alpha_1, ..., alpha_n],[beta_0, beta_1, ..., beta_n]], alpha_i et beta_i pénalisant respectivement les faux positifs et les faux négatifs de la classe i.
    * Pour la *FocaleLoss*, **loss_weights** doit être de la forme [w_0, w_1, ..., w_n], w_i étant le poids associé à la
      i_ème classes.

* **weights_name**: le nom du fichier contenant les poids du modèle qui sera enregistré dans le dossier indiqué dans la ligne de commande de l'entrainement.

```yaml
model_name: MinkUNet14 #Voir readme pour les possibilités
voxel_size: 0.08 #Taille des voxels en mètres.
batch_size: 2
test_batch_size: 1
epochs: 1000
patience: 100
num_worker: 1
pin_memory: False
learning_rate: 1e-4
weight_decay: 1e-5
self_attention: 0 # 0, 1 ou 2 couches
multihead_attention: 1 # 0, 1 ou 2 couches.
num_head: 4 #Nombre de têtes de la MultiHeadAttention
loss: FocalTverskyLoss_multiclass #Choix: FocalTverskyLoss (cas binaire), FocalLoss (tout cas de figure), FocalTverskyLoss_multiclass (> à 2 classes)
loss_weights: [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]] #Voir le Readme pour plus d'info selon la loss souhaitée.
weights_name: 'eclair' #nom des poids qui seront sauvegardés
```

<br>

#### Transformations appliquées aux nuages

Un certain nombre de transformations sont appliquées aux nuages avant d'être placés en entrée du modèle. 

* **cloud_size** recadre les points du nuage dans une boîte 3D de taille fixe et filtre
  les points aberrants (bruit sous le nuage) selon un quantile sur l’axe Z. Cette transformation uniformise l’échelle spatiale des données
  pour
  faciliter leur traitement par le modèle. Si les nuages sont plus petits que la taille indiquée, un point fictif sera
  ajouté (puis supprimé à la fin, don't worry).
  Si les nuages sont plus grands, les points au dela des limites définies ne seront pas pris en compte pour l'entrainement
  et seront classés à zéro pour la prédiction.

Des transformations peuvent être appliquées ou non aux données pour améliorer l'entrainement et sont à indiquer dans le paramètre **train_transforms** :

* Onze méthodes de data-augmentation sont disponibles. Il suffit de commenter celles non souhaitées pour ne pas les appliquées :
  * *Intensity_RGB_Variation* : Applique une variation aléatoire aux valeurs d'intensité et RGB (si disponibles) entre de ±15%. Les valeurs RGB sont limitées à une plage valide (par défaut 65535 pour 16 bits).
  * *Scale* : Applique une variation aléatoire de ±10 % aux coordonnées XYZ du nuage de points.
  * *Noise* : Ajoute un bruit aléatoire uniforme entre -5 % et +5 % aux champs intensité et RGB (si disponibles). Les valeurs RGB sont limitées à une plage valide (par défaut 65535).
  * *Rotate* : Applique une rotation aléatoire autour de l'axe Z (angle entre 0 et 2π) aux coordonnées du nuage de points.
  * *Flip* : Inverse les coordonnées Y avec une probabilité de 50 % (pas d'inversion sur l'axe X dans la version actuelle).
  * *RandomCrop* : Découpe aléatoirement une portion du nuage de points, en sélectionnant une boîte englobante dont les dimensions varient entre 90 % et 100 % de la taille du nuage.
  * *RandomDelete* : Supprime aléatoirement entre 0,1 % et 10 % des points du nuage, en mettant à jour les attributs associés (intensité, RGB, classification, etc.).
  * *AddRandomPoints* : Ajoute un nombre aléatoire de points (jusqu'à 0,001 % du total) avec des coordonnées dans une boîte englobante légèrement étendue autour du nuage. Les nouveaux points ont des valeurs RGB aléatoires (0 à 65535) et une classification fixée à 0.
  * *ElasticDistortion* : Applique une déformation élastique en générant un champ de déplacements aléatoires lissé par un filtre gaussien. Les paramètres sigma (0,01 à 0,3) et scale (0,01 à 0,1) contrôlent l'amplitude de la déformation.
  * *ShearTransform* : Applique un cisaillement aléatoire aux coordonnées avec une amplitude entre 0,001 et 0,01, modifiant les relations entre les axes X, Y et Z.
  * *StretchTransform* : Applique un étirement ou une compression aléatoire indépendante sur chaque axe (X, Y, Z) avec des facteurs variant dans un intervalle aléatoire (par défaut entre 0,85 et 1,15).

  
* 3 méthodes de mise en forme des données (déconseillé d'y toucher), utilisées pour les données en train, val et test :
    - *NormalizeCoordinates* / *NormalizeCoordinates_predict* : normalise les coordonnées et recadre les points dans la boite englobante définie.
    - *NormalizeFeatures* : normalise les features.
    - *RemapClassification* : réordonne les ids à partir des valeurs indiquées dans "input_mapping".

```yaml
cloud_size: #Taille des nuages fixe. Voir readme pour plus d'info.
  max_range_xy: 40.0
  max_range_z: 40.0

#Data-augmentation appliquées. Plus d'info dans le readme. Il suffit de commenter pour ne pas appliquer l'augmentation.
train_transforms:
  - name: Rotate
    params:
      feature_names: ${feature_names}
  - name: Flip
    params:
      feature_names: ${feature_names}
  - name: Scale
    params:
      feature_names: ${feature_names}
  - name: RandomDelete
    params:
      feature_names: ${feature_names}
#  - name: RandomCrop
#    params:
#      feature_names: ${feature_names}
#  - name: AddRandomPoints
  - name: ElasticDistortion
  - name: ShearTransform
  - name: StretchTransform
  - name: Intensity_RGB_Variation
    params:
      feature_names: ${feature_names}
  - name: Noise
    params:
      feature_names: ${feature_names}
  - name: NormalizeCoordinates
  - name: NormalizeFeatures
    params:
      feature_names: ${feature_names}
  - name: RemapClassification
    params:
      class_mapping: ${input_mapping}


###Normalisation de base pour les nuages de test et validation. Il est déconseillé d'y toucher.
val_transforms:
  - name: NormalizeCoordinates
  - name: NormalizeFeatures
    params:
      feature_names: ${feature_names}
  - name: RemapClassification
    params:
      class_mapping: ${input_mapping}

test_transforms:
  - name: NormalizeCoordinates_predict
  - name: NormalizeFeatures
    params:
      feature_names: ${feature_names}
  - name: RemapClassification
    params:
      class_mapping: ${input_mapping}
```

#

#### Entrainer le modèle

Une fois que tout est en place, il suffit d'exécuter la commande : 

```bash
python scripts/train.py --data_dir [the directory path where the pointclouds are stored (train & val)] --label_json [path to the.json that specifies which set (train, validation, or test) each cloud belongs to] --config_yaml [path to the config.yaml file] --save_dir [the directory path where the weights will be stored]
```

* Les fichiers poids sauvegardés ont tous un nom de la forme : **weights_name**_[Type de poids].pth
* [Type de poids] peut prendre la forme de :
    * "final_weights" pour les poids de la dernière epoch.
    * "bestvallossweights" pour les poids minimisant la validation loss
    * "bestvalaccweights" pour les poids maximisant la validation accuracy
    * "bestvalf1scoreweights" pour les poids maximisant le F1 score de la classe numéro 1 (uniquement dans le cas
      d'une classification binaire)

<br>

## Prédire des nuages de points

La prédiction de nuages de points s'effectue avec le script ```scripts/predict.py```. Le script utilise le même fichier de configuration que pour l'entrainement.

Ce script se lance avec la commande :

```bash
python predict.py --weight_path [path to the model weights] --config_file [path to the config.yaml file used for training] --pointclouds [either .json file or path to a folder with pointclouds to predict] --savepath [path to the save folder]
```

Les nuages indiqués en "test" dans le .json seront alors prédits, ou bien tous les nuages du dossier, selon ce qui est
renseigné dans l'argument ```--pointclouds```

Les prédictions sont enregistrées dans un nouveau champ **eclair**.

## Intégration dans PSANP

Ce projet implémente un [plugin pour PSANP](http://geosat-docs.s3-website.eu-west-3.amazonaws.com/psanp-abstractions/master/plugins.html),
ce qui permet d'enchaîner une classification d'Eclair avec d'autres types de traitements
(sous-échantillonnage, génération d'ortho-images...). Pour que PSANP puisse charger 
les poids et la configuration d'un modèle, la structure suivante doit être respectée :

```
<classifiers>
├───EclairBaseClassifier
│       model.yaml
│       weights.pth
│
├───EclairPLOClassifier
│       model.yaml
│       weights.pth
│
└───...
```

**Notes**

- Dans l'exemple ci-dessus, les dossiers `EclairBaseClassifier` et `EclairPLOClassifier` sont appelés *bundles*.
- Ils **doivent** contenir un fichier de configuration d'*Eclair* `model.yaml` et un fichier de poids `weights.pth`.
- Leur nom **doit** commencer par `Eclair` et finir par `Classifier`, et suivre la convention *PascalCase*.
- Pour que PSANP puisse charger les bundles, la variable d'environnement 
  `PSANP_BUNDLE_FOLDER` doit pointer vers le dossier `<classifiers>`

> ℹ️ Le script `scripts/check_psanp_bundles.py` permet de vérifier si un bundle est valide pour PSANP 

---

## En savoir plus

* Le repo git du projet MinkowkiEngine utilisé par Eclair est disponible ici: https://gitlab.geo-sat.com/rnd/research/minkowski-engine 
* Les rapports de recherches sur Eclair sont disponibles ici: `R:\4_archived_projects\4_reports\1_research_reports\Eclair`