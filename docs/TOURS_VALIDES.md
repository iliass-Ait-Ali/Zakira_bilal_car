# Validation du pilote déterministe — 15 novembre 2026

Note de suivi : ce rapport décrit l'étape de validation des tours. Depuis, [le dataset a été audité et réservé exclusivement à `baseline`](DATASET_BASELINE.md) ; les anciennes mentions d'une source manuelle par défaut ci-dessous décrivent l'état historique.

**Résultat : quatre tours complets validés sur `roboracingleague_1`, dont trois consécutifs pendant un essai autonome de 180 secondes.** Aucun entraînement CNN. Aucun changement de son architecture. Le simulateur déjà ouvert a été utilisé sur `127.0.0.1:9091`, puis chaque client a été arrêté proprement avec une commande de freinage.

## Essais réellement exécutés

Les quatre essais ci-dessous sont ceux de cette mission. L'ancien essai utilisateur `924fc87...` a été analysé, mais n'est pas compté comme un nouvel essai.

| Essai / dossier sous `runs/lap_checks` | Conduite effective, horloge locale | Résultat | Tours validés |
|---|---:|---|---:|
| `b7b4805b14e046918a2a24401a88929a` | 3,10 s | Reproduction instrumentée : watchdog déclenché ; 2,70 s couvertes par les décisions | 0 |
| `7d95624cf6e44c9ab38574684d07b1f8` | 30,00 s | Délais corrigés ; ancien guidage, dérive près du mur clair, CTE maximal 3,108 | 0 |
| `2dbda007dd8c46dba65ca0c3a4cd86ba` | 65,00 s | Nouveau guidage et régulation de vitesse ; CTE maximal 0,552 | 1 : **56,33 s** |
| `885e5dd421a94306ae7722714fe75a16` | 180,00 s | Confirmation continue ; CTE maximal 0,556 | 3 : **56,36 / 56,14 / 56,11 s** |

Les durées de conduite commencent après disponibilité de la caméra. Le chargement ajoute environ une seconde. Le dernier essai couvre 179,95 s entre sa première et sa dernière décision enregistrée ; l'arrêt normal intervient à 180 s. Les temps au tour sont les différences des `timeStamp` des événements de ligne, pas des estimations à partir de la durée d'exécution.

Aucune collision `hit` n'a été signalée dans ces quatre essais. Le deuxième essai a montré une dérive importante vers le bord gauche aux nœuds 72–74 : il n'est pas retenu comme démonstration fiable. Les essais réussis ne présentent ni sortie visible dans les captures examinées, ni anomalie de trajectoire, ni arrêt pendant les tours. Le CTE est exprimé dans les unités du simulateur ; sa limite conservatrice de 1,0 n'est pas présentée comme une mesure calibrée de largeur de piste.

## Preuves des tours

Le validateur `racing/laps.py` exige simultanément :

- Deux passages successifs sur la ligne **d'indice 0**. La ligne d'indice 1 observée à mi-parcours n'est pas comptée comme un tour.
- Les **147 nœuds distincts**, parcourus dans l'ordre, sans saut ni marche arrière ; progression nette de 147 nœuds pour chacun des quatre tours validés.
- Retour à la même position : écart entre observations proches des deux franchissements inférieur à 0,3 unité de position. Sur les trois tours finaux : **0,037 / 0,034 / 0,067**.
- Absence de collision, de commande manuelle, d'observation périmée, d'arrêt en piste et de dépassement du CTE conservateur de 1,0.

La vitesse minimale des deuxième et troisième tours finaux est 0,786 et 0,784. Le premier tour comprend l'accélération initiale et descend à 0,532 ; la voiture ne s'arrête pas. Aucun guidage par nœuds, positions ou CTE n'est utilisé : le pilote emploie l'image et la vitesse. Ces autres champs servent à la vérification et au filtrage des futurs labels.

Preuves consultables :

- [Validation détaillée des trois tours](../runs/lap_checks/885e5dd421a94306ae7722714fe75a16/lap_validation.json).
- [Décisions, télémétrie et événements bruts](../runs/lap_checks/885e5dd421a94306ae7722714fe75a16/decisions.jsonl).
- [Trajectoire, progression des nœuds et CTE](../runs/lap_checks/885e5dd421a94306ae7722714fe75a16/trajectory.png).
- [Captures réelles de la caméra](../runs/lap_checks/885e5dd421a94306ae7722714fe75a16/camera_evidence.png).
- [Premier tour de 56,33 s](../runs/lap_checks/2dbda007dd8c46dba65ca0c3a4cd86ba/lap_validation.json).
- [Mesures consolidées et empreintes SHA-256 des journaux](lap_evidence.json).

## Cause et correction des ralentissements

L'ancien journal `runs/driving/924fc87bf98b4fcfbc07d77b0a5e3f77` montre 166 paquets de télémétrie, aucune erreur réseau/protocole, puis `TelemetryError: stale_frame` à 9,04 s après connexion. Il ne contient pas de chronométrage par étape : il ne permet donc pas d'attribuer rétrospectivement chaque milliseconde de cet arrêt.

La reproduction instrumentée a révélé un coût important de la réparation des nombres décimaux français. L'expression régulière parcourait les chaînes base64 caractère par caractère, avec alternatives répétées et blocage du GIL Python. Le décodage JSON prenait en moyenne **38,84 ms** et jusqu'à **53,87 ms** dans cette reproduction. Sur un paquet construit avec une vraie capture de 107 018 octets, l'ancien réparateur plus `json.loads` prenait **50,59 ms en moyenne sur 100 passages**. Ces blocages s'ajoutaient au décodage PNG, à l'affichage synchrone et aux captures ; le watchdog a effectivement arrêté la voiture.

Corrections : consommation des chaînes par blocs dans l'expression régulière, affichage et captures dans un thread distinct avec une seule image en attente, calcul visuel limité à la bande utile de l'image. Les diagnostics mesurent désormais chaque étape. L'horodatage de réception est pris avant le décodage JSON, afin d'inclure son coût dans l'âge de décision.

Mesures du dernier essai, 3 600 images décodées et 3 501 commandes :

| Mesure | Moyenne | 95e percentile | Maximum |
|---|---:|---:|---:|
| Intervalle entre télémétries | 50,00 ms | 52,00 ms | 90,68 ms |
| Décodage JSON / réparation locale | 2,13 ms | 2,93 ms | 3,62 ms |
| Décodage image | 3,33 ms | 4,20 ms | 7,57 ms |
| Calcul du pilote | 2,26 ms | 3,13 ms | 24,16 ms |
| Réception du message → commande | 29,43 ms | 50,23 ms | **56,37 ms** |

Cette dernière durée inclut l'attente de la prochaine échéance de commande à 20 Hz. Elle ne prétend pas mesurer la latence de rendu Unity avant émission. Le seuil d'image récente reste **150 ms**, et le watchdog reste **400 ms**. Aucune image périmée n'est utilisée. Une image transitoire 160×120 est rejetée pendant la configuration ; toutes les images de conduite sont 255×255. Aucune erreur réseau, aucun `stale_frame` ni watchdog après correction lors des essais de 30, 65 et 180 s.

## Paramètres du pilote retenu

Le nouveau `BaselinePilot` détecte des segments blancs fins, en rejetant les grandes surfaces claires : luminance minimale 180 par canal, écart entre canaux inférieur à 65, largeur des segments au plus 5,5 % de l'image. Il examine une ligne sur deux entre 51 % et 65 % de la hauteur, choisit les limites proches du centre et utilise la médiane des centres. Moins de quatre lignes exploitables provoquent le freinage.

La médiane supprime l'influence excessive des panneaux, murs et marques transversales. Cette bande fournit une visée devant la voiture ; il n'y a pas de reconstruction géométrique ou de prédiction explicite de courbure.

Paramètres dans `configs/race.json`, rubrique `baseline` :

| Paramètre | Valeur |
|---|---:|
| Gain de direction | 10 |
| Constante de filtrage | 0,06 s |
| Vitesse maximale de changement de direction | 5 unités/s |
| Consigne ligne droite / braquage maximal | **1,0 / 0,7** unité de vitesse télémétrique |
| Accélérateur maximal / terme constant | 0,18 / 0,08 |
| Gains de vitesse proportionnel / intégral | 0,35 / 0,08 |

La consigne diminue avec la demande de braquage. Au-delà d'un excès de vitesse de 0,15, l'accélérateur est nul et le frein vaut au plus 0,5, avec gain 0,4. Accélérateur et frein ne sont jamais positifs simultanément. Les anciens `runtime.baseline_gain` et `runtime.baseline_throttle` ne servent qu'à `legacy_baseline`, conservé pour comparaison ; le mode CLI `baseline` utilise `BaselinePilot`.

## Vérification de la collecte CNN

Seulement **40 images étiquetées**, couvrant 2 s, ont été enregistrées sous `runs/expert_recording_check/362e462e1a154addb8b9a52b360a00fe`. Ce dossier est distinct des données utilisateur existantes. Les captures de diagnostic sont séparées : 273 images au total sur les quatre essais, sans constitution d'un dataset d'entraînement.

Les 40 PNG sont uniques, lisibles, RGB 255×255. Les commandes, vitesses, télémétries et horodatages correspondent exactement aux décisions portant le même `frame_id`. L'âge maximal des décisions enregistrées est 54,23 ms. `load_sessions` accepte les 40 exemples sans rejet lorsque `baseline` est explicitement autorisé comme source ; les fenêtres et labels futurs sont construits correctement. [Rapport de lecture](../runs/expert_recording_check/verification.json).

`--record-limit 40` borne l'enregistrement sans arrêter la conduite. Le marqueur `--baseline-validated` reste explicite. Les futurs labels baseline sont invalidés si les lignes sont perdues, si une collision est signalée ou si |CTE| dépasse 1. La configuration d'entraînement reste par défaut limitée à `manual` ; une future configuration de collecte/préparation pourra autoriser `["manual", "baseline"]` après revue de ses sessions.

Ces 40 exemples vérifient le fonctionnement de l'enregistrement ; ils ne constituent pas une base d'entraînement suffisante. La robustesse constatée concerne cette piste, cette caméra et ce départ. La diversité de positions, de vitesses et de récupérations reste à collecter ultérieurement. **Aucune préparation de dataset massif ni aucun entraînement n'a été lancé.**

## Reproduire

Depuis la racine du projet, simulateur ouvert sur 9091 :

```powershell
.\.venv\Scripts\python.exe -m racing.drive --mode baseline --connect --display --seconds 180 --snapshots 1.6 --logdir runs/lap_checks
```

Puis utiliser le dossier affiché en fin d'essai :

```powershell
.\.venv\Scripts\python.exe -m racing.laps runs/lap_checks/IDENTIFIANT --plot
```

La première commande ne collecte pas de dataset. Les captures sont plafonnées à 120 par essai. Pour reproduire uniquement la vérification bornée de l'enregistrement, ajouter `--record --record-limit 40 --baseline-validated --group expert_smoke_check --sessions runs/expert_recording_check`.

## Fichiers et vérifications

Code modifié : `racing/baseline.py`, `config.py`, `drive.py`, `session.py`, `transport.py`, `diagnostics.py`, `vendor/client.py`, `vendor/util.py`, `vendor/NOTICE.md`, `configs/race.json`.

Code ajouté : `racing/preview.py`, `racing/laps.py`, `tests/test_laps.py`, `tools/lap_evidence.py`. Documentation : présent rapport, `docs/lap_evidence.json`, README et liens de suivi.

**42 tests passent**, dont le rejet d'une image périmée après optimisation, le freinage en perte de ligne, la résistance au mur clair, et le refus de faux tours fondés sur un événement isolé, des nœuds manquants, une collision ou une intervention manuelle. Les fichiers originaux, le simulateur et l'architecture CNN sont préservés.
