# Dataset exclusivement baseline — 15 novembre 2026

L'audit porte sur les cinq sessions réellement présentes dans `data/sessions`, et non sur les captures de diagnostic ni les 40 images du précédent contrôle technique conservées sous `runs`. Le CNN n'est pas entraîné. Son architecture reste inchangée.

## Acquisition et qualité

Les cinq sessions contiennent **29 199 images et commandes**, toutes de source `baseline`, réparties sur cinq groupes `auto_01` à `auto_05`. Chaque session couvre environ 300 secondes : environ 25 minutes au total. Les fichiers image originaux et leurs journaux sont préservés.

| Groupe | Session | Images brutes | Tours vérifiés | CTE absolu maximal |
|---|---|---:|---:|---:|
| auto_01 | f3780f16e77941b9a60f8f027cec533b | 5 837 | 5 | 0,570 |
| auto_02 | 0a3231f860c04ea3a7bf46a1f6dd3166 | 5 828 | 5 | 0,560 |
| auto_03 | e08296db21c940518383c3c91841dd41 | 5 845 | 5 | 0,558 |
| auto_04 | d06313c6d08045ec91605c06cb891886 | 5 855 | 5 | 0,562 |
| auto_05 | 4776bf6c76844958a2e09f4096bbcf16 | 5 834 | 5 | 0,550 |

Les groupes correspondent à des exécutions distinctes : cinq identifiants de séquence différents, intervalles d'acquisition non superposés, remises à zéro du temps du simulateur, et cinq journaux de conduite correspondants sans faute terminale. **Les 25 tours ne constituent pas 25 groupes.** Chaque groupe reste entier dans un seul split.

Contrôles exhaustifs effectués sur les enregistrements :

- 29 199 PNG décodés, tous **RGB 255×255**, aucun fichier manquant ou orphelin.
- `steering`, `throttle` et `brake` présents, finis, dans leurs bornes ; accélérateur et frein mutuellement exclusifs. **595 images comportent une commande de freinage**.
- Vitesse finie et conforme au champ télémétrique ; temps du simulateur cohérent ; horloges réception/envoi ordonnées et compatibles avec `decision_age_s` ; numéros d'images croissants.
- Aucun intervalle inter-image supérieur à 150 ms ; délai décisionnel maximal **79,40 ms**, sous la limite de 150 ms.
- Aucun enregistrement manuel, CNN, sans label valide, en collision, en perte de ligne ou au-delà du CTE conservateur de 1,0.
- Correspondance exacte avec les journaux de conduite pour le numéro de frame, la commande, la source, la vitesse, les horodatages et la télémétrie : **aucune divergence**.
- **25 tours vérifiés** à partir des événements de ligne et des 147 nœuds dans l'ordre. Les images de virage et d'écart maximal de chaque session ont été examinées. Aucune sortie visible dans ces captures ni anomalie de progression. Le CTE n'est pas une mesure calibrée de la largeur physique de la piste.

[Audit technique brut, avant exclusion des doublons](../data/audit/baseline_v1/audit.json) · [Échantillons examinés](../data/audit/baseline_v1/sample_review.png).

## Exemples exclus

Les 29 199 enregistrements sont techniquement valides. Trois images distinctes, pixel pour pixel, apparaissent néanmoins dans plusieurs groupes au démarrage. Elles représentent **8 enregistrements** : images `00000000.png` et `00000001.png` dans `auto_01`, `auto_02`, `auto_04` et `auto_05`.

La préparation exclut les huit occurrences de tous les groupes concernés, avant de construire les horizons futurs. Il ne s'agit pas de données manuelles ni d'images corrompues. Les originaux ne sont pas supprimés. Toute exclusion coupe les fenêtres temporelles ; aucun label futur n'est prolongé à travers une image rejetée.

La règle empêche aussi l'inclusion future de sources non baseline, de commandes incohérentes, d'images invalides, d'observations périmées, de collisions, de pertes de lignes ou de CTE excessifs. Une même séquence physique attribuée à plusieurs groupes provoque un refus de préparation.

## Dataset préparé

`racing.data` a effectivement créé **`data/prepared/baseline_v1`**, avec **29 191 exemples exploitables**, tous de source `baseline`. Le tirage des groupes est reproductible, graine 42 :

| Split | Groupes entiers | Images / exemples | Trois horizons disponibles |
|---|---|---:|---:|
| Entraînement | auto_01, auto_02, auto_04 | **17 514** | 17 487 |
| Validation | auto_05 | **5 832** | 5 823 |
| Test | auto_03 | **5 845** | 5 837 |
| Total | **5 groupes** | **29 191** | **29 147** |

Les 44 exemples restants sont conservés avec un masque sur les horizons indisponibles en fin de session ; leur direction actuelle reste supervisée. Aucun groupe, aucune séquence, aucun chemin d'image ni aucune image identique pixel pour pixel ne sont partagés entre les splits. Il n'y a pas de mélange aléatoire de frames ni de séparation par tour.

Fichiers produits : [manifeste et empreintes](../data/prepared/baseline_v1/dataset.json), [liste des 8 exclusions](../data/prepared/baseline_v1/rejected.json), [bilan final](../data/prepared/baseline_v1/audit_summary.json), `train.jsonl`, `val.jsonl`, `test.jsonl`.

Le rapport d'audit brut conserve les nombres avant déduplication (29 199). Le manifeste préparé et le bilan final donnent les nombres réellement destinés au CNN (29 191). Les SHA-256 des métadonnées et enregistrements sources ont été revérifiés après préparation : ils n'ont pas changé.

## Ce que le CNN apprend

L'image RGB est découpée puis redimensionnée en **3×96×160**, normalisée dans [−1, 1]. La vitesse, divisée par 10, constitue une entrée supplémentaire. Les trois cibles sont les directions à **t, t+0,25 s et t+0,50 s**, avec un masque pour les futurs indisponibles en fin de séquence.

L'accélérateur et le frein restent présents dans les enregistrements audités ; ils ne deviennent pas des sorties supplémentaires du CNN. L'architecture de **232 751 paramètres** et ses trois sorties de direction sont conservées. La commande longitudinale demeure calculée par le contrôleur.

La provenance `baseline` est inscrite dans chaque exemple préparé et dans le manifeste. Avant entraînement ou évaluation, les trois splits sont contrôlés : sources, empreintes, sessions, groupes, séquences, chemins d'images, empreintes des pixels et dimensions des labels. Une ancienne préparation incluant `manual`, ou dépourvue de provenance, est refusée.

La vérification `racing.check --dataset` contrôle tous les exemples préparés puis réalise un passage avant sur 64 images réelles réparties dans chacun des trois splits. Elle ne crée pas d'optimiseur, ne rétropropage pas et ne sauvegarde pas de poids entraînés. [Résultat du contrôle de compatibilité](../data/prepared/baseline_v1/compatibility.json).

## Suffisance et diversité

Ce jeu est suffisant pour **un premier entraînement d'imitation sur ce circuit et cette caméra**, avec un vrai découpage par acquisition. Il contient les 147 nœuds dans chaque groupe, des virages des deux signes, des lignes droites, l'accélération de départ et du freinage. La vitesse médiane est 0,840 ; 90 % des observations sont entre environ 0,790 et 1,006.

La diversité reste limitée : même circuit, même caméra, mêmes consignes 1,0/0,7, trajectoires centrées proches et aucune récupération importante enregistrée. Le nombre de frames ne mesure pas le nombre de situations indépendantes. Un bon score de test mesurera d'abord la reproduction de ces conditions ; il ne prouvera ni la robustesse aux départs décentrés, ni la tenue de piste autonome du CNN.

**Sessions supplémentaires nécessaires pour lancer cette première référence : 0.** Pour élargir ensuite les conditions, **5 sessions supplémentaires de 300 s** sont un lot raisonnable, portant le total à 10 groupes. Le script suivant est prêt mais n'a pas été exécuté ; simulateur ouvert sur 9091 :

```powershell
powershell -NoProfile -File tools/collect_baseline.ps1 -FirstGroup 6
```

Il crée `auto_06` à `auto_10`, avec les couples de consignes ligne droite/virage **0,85/0,60 ; 0,90/0,65 ; 0,95/0,65 ; 1,00/0,70 ; 0,90/0,60**. Il évite la réutilisation d'un groupe, lance uniquement le pilote déterministe et exige au moins trois tours vérifiés par session. Ces variantes plus lentes restent à valider en simulation. Elles améliorent la diversité de vitesse, sans garantir des exemples de récupération latérale. Aucun entraînement ne suit automatiquement.

Commande élémentaire si l'on souhaite une seule session supplémentaire au réglage déjà éprouvé :

```powershell
.\.venv\Scripts\python.exe -m racing.drive --mode baseline --connect --record --baseline-validated --group auto_06 --seconds 300
```

Tous les tours d'une même exécution gardent le même groupe. Pour un futur dataset `baseline_v2`, conserver le groupe de test initial hors du réglage des hyperparamètres ; ne pas relancer aveuglément un tirage aléatoire après avoir utilisé son score.

## Entraînement à lancer personnellement

Après la vérification du dataset préparé :

```powershell
.\.venv\Scripts\python.exe -m racing.train --config configs/race.json --dataset data/prepared/baseline_v1 --out runs/cnn_baseline_v1 --start-training
```

Cette commande n'a pas été exécutée. Elle prévoit au maximum 80 époques, batch 64, AdamW à 3e-4, décroissance cosinus et arrêt anticipé après 12 époques sans amélioration suffisante. Le test ne participe ni à l'optimisation ni au choix du meilleur checkpoint.

Après entraînement :

```powershell
.\.venv\Scripts\python.exe -m racing.check --checkpoint runs/cnn_baseline_v1/best.pt --dataset data/prepared/baseline_v1
.\.venv\Scripts\python.exe -m racing.check --checkpoint runs/cnn_baseline_v1/last.pt
Get-FileHash -Algorithm SHA256 runs/cnn_baseline_v1/best.pt
Get-Content runs/cnn_baseline_v1/checkpoint_hashes.json
.\.venv\Scripts\python.exe -m racing.plot_metrics runs/cnn_baseline_v1/metrics.csv --out runs/cnn_baseline_v1/metrics.png
```

Vérifier des poids et pertes finis, un nombre d'époques positif, les 232 751 paramètres, une sortie `[N,3]`, la concordance des empreintes et du contrat caméra. `best.pt` correspond à la meilleure perte de validation ; `last.pt` à la dernière époque exécutée. Une loss train qui baisse tandis que la loss validation remonte indique un surapprentissage possible. Suivre aussi `val_mae` et `val_turn_mae`, pas seulement la moyenne globale.

Une fois le checkpoint choisi sans consulter le test :

```powershell
.\.venv\Scripts\python.exe -m racing.evaluate offline --checkpoint runs/cnn_baseline_v1/best.pt --dataset data/prepared/baseline_v1 --out runs/cnn_baseline_v1/test_metrics.json
```

Les erreurs hors ligne ne remplacent pas un futur essai en boucle fermée. Aucun checkpoint entraîné ni courbe d'apprentissage réelle n'a été inventé ou créé pendant cet audit.

## Modifications et vérification du code

La configuration `configs/race.json` autorise uniquement `baseline`. `racing/data.py` applique les contrôles de qualité, coupe les séquences invalides, déduplique entre groupes et inscrit la provenance. `racing/train.py` et `evaluate.py` vérifient cette provenance dans les trois splits. `racing/check.py` accepte un dataset réel sans démarrer d'entraînement. `racing/audit_data.py` fournit l'audit reproductible.

`racing/drive.py` refuse l'enregistrement manuel/CNN et neutralise la reprise au clavier dans les modes autonomes ; `preview.py` adapte le titre de l'aperçu. `tools/collect_baseline.ps1` prépare la collecte automatique facultative. Le README et la procédure d'évaluation ont été mis à jour. `racing/model.py`, le prétraitement et les sessions sources n'ont pas été modifiés.

**48 tests passent**, dont le refus des sources manuelles même dans un manifeste réempreinté, le contrôle des trois splits, l'exclusion des pixels identiques entre groupes, la coupure des labels futurs et la détection d'une même séquence répartie dans plusieurs groupes. Le script PowerShell a été vérifié syntaxiquement, sans être exécuté.
