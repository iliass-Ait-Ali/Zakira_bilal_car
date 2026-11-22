# Zakira_bilal_car

## État actuel (21 novembre 2026) : le projet est fini

Une voiture du simulateur DonkeyCar roule **seule sur les 10 circuits**, conduite par un **réseau de neurones (CNN)** qui ne voit que la **caméra** et sa vitesse. Le modèle final est **v14** ([runs/cnn_all10_v14/best.pt](runs/cnn_all10_v14/best.pt)).

**📘 Pour tout comprendre, lire [docs/RAPPORT_COMPLET.md](docs/RAPPORT_COMPLET.md).** Il explique, depuis les bases, comment tout a été fait, et pourquoi.

| Mode de conduite | Essais réussis (2 minutes, aucune collision, pas de sortie de route) |
|---|---|
| **100 % réseau** (volant, accélérateur et frein appris) : configs `configs/tracks/cnn_v14n_<circuit>.json` | 10/10 puis 8/10, soit **18 essais sur 20** |
| **100 % réseau**, démonstration du 22 novembre ([results/final_cnn_all10_v14_demo.jsonl](results/final_cnn_all10_v14_demo.jsonl)) | **9/10** : aucune collision ; sur sparkfun_avc, 2,5 s à 33 cm de la ligne (limite 30 cm) |
| Volant appris, vitesse par une petite règle : configs `configs/tracks/cnn_v14_<circuit>.json` | **10/10** |

### Essayer en 3 clics (Windows, sans carte graphique)

1. Télécharger **`Zakira_bilal_car_court.zip`** (239 Mo : modèle v14 + simulateur) dans la page **[Releases](https://github.com/iliass-Ait-Ali/Zakira_bilal_car/releases)**, puis le dézipper.
2. Installer Python 3.12 ou 3.13 ([python.org](https://www.python.org/downloads/), cocher « Add python.exe to PATH »), puis double-cliquer **`INSTALLER.bat`** (une seule fois).
3. Double-cliquer **`LANCER_VOITURE.bat`** et choisir un circuit de 1 à 10 : la voiture roule seule 2 minutes.

Les instructions sont aussi dans [LISEZMOI.txt](LISEZMOI.txt).

### Pour aller plus loin (projet complet)

Faire conduire le modèle final sur un circuit. Le simulateur s'ouvre et se ferme tout seul ; Git Bash est nécessaire.

```bash
bash tools/sim_run.sh runs/cnn_drive cnn -- --mode cnn --checkpoint runs/cnn_all10_v14/best.pt --config configs/tracks/cnn_v14n_waveshare.json --seconds 120
```

Tester les 10 circuits, puis vérifier collisions et sorties de route :

```bash
bash tools/final_eval.sh runs/cnn_all10_v14/best.pt cnn_v14n 120 essai1
.venv/Scripts/python.exe tools/offtrack_check.py results/final_cnn_all10_v14_essai1.jsonl
```

Pour l'installation, l'entraînement et la collecte, voir la partie 17 du rapport complet. Toutes les décisions, avec leurs chiffres, sont dans [docs/JOURNAL_AUTONOME.md](docs/JOURNAL_AUTONOME.md).

### Ce qui n'est pas sur GitHub

GitHub n'accepte pas les très gros dossiers. Ces parties sont seulement dans le dossier `Zakira_bilal_car` partagé à part :

- `data/sessions/` : les 64 enregistrements des professeurs, soit 29 Go d'images. Sans eux, on peut faire conduire les modèles, mais pas les réentraîner.
- `DonkeySimWin/` : le simulateur, 0,6 Go. Il se télécharge aussi depuis le projet officiel sdsandbox / gym-donkeycar, version Windows.
- `src/donkeycar/` : copie du projet officiel [DonkeyCar](https://github.com/autorope/donkeycar).
- les journaux de conduite des tests (`runs/final_eval`, `runs/driving`). Les résumés sont dans `results/`.

Les modèles principaux sont sur GitHub (`runs/cnn_all10_v14/best.pt`, etc.).

Le reste de ce document décrit la mise en place d'origine. Il est conservé tel quel.

---

Le projet est préparé pour apprendre une conduite supervisée sur `roboracingleague_1`.
**Le dataset baseline est prêt : 29 191 exemples, répartis en 17 514 entraînement / 5 832 validation / 5 845 test, dans cinq groupes séparés.** Aucune donnée manuelle n'est autorisée. Aucun entraînement du CNN n'a été lancé. Voir [le bilan précis et la commande d'entraînement](docs/DATASET_BASELINE.md). La suite compte 48 tests réussis.

Le pilote déterministe avait d'abord validé quatre tours, dont trois consécutifs en 180 secondes : 56,36 / 56,14 / 56,11 s. Voir [les preuves de conduite](docs/TOURS_VALIDES.md). Les cinq sessions collectées ensuite ont fourni 25 tours supplémentaires vérifiés pendant l'audit.

Pour reproduire la conduite autonome validée, simulateur ouvert sur le port 9091 :

```powershell
.\.venv\Scripts\python.exe -m racing.drive --mode baseline --connect --display --seconds 180 --snapshots 1.6 --logdir runs/lap_checks
```

Puis vérifier le dossier indiqué en fin d'essai avec `python -m racing.laps runs/lap_checks/IDENTIFIANT --plot`. Les captures sont limitées à 120 ; cette commande de conduite ne collecte pas de dataset.

Pour le problème de démarrage, consulter [la correction et ses preuves réelles](docs/CONNEXION_SDSANDBOX.md). Diagnostic sans accélération :

```powershell
.\.venv\Scripts\python.exe -m racing.drive --diagnose --connect --debug
```

Les 336 anciennes images de `src/camera`, en 160×120 sans labels ni horodatages, restent exclues. Le dataset préparé utilise exclusivement les nouvelles sessions `baseline` en RGB 255×255, avec leurs commandes enregistrées.

## Ce qui est prêt

| Fichier | Rôle |
|---|---|
| `racing/model.py` | CNN de 232 751 paramètres, cinq convolutions, vitesse en entrée et trois directions en sortie |
| `racing/preprocess.py` | Conversion RGB, découpe, redimensionnement et normalisation communs |
| `racing/drive.py` | Collecte manuelle, pilotage CNN, reprise manuelle et pilote de référence |
| `racing/baseline.py`, `racing/laps.py`, `racing/preview.py` | Expert visuel déterministe validé, preuves des tours et affichage séparé du contrôle |
| `racing/transport.py`, `racing/vendor/` | `SDClient` corrigé : framing, réception asynchrone, écritures partielles, arrêt des threads |
| `racing/session.py`, `racing/diagnostics.py`, `racing/diagnose.py` | Démarrage menu/piste, diagnostic détaillé et test caméra sans accélération |
| `racing/recording.py` | Enregistrement synchronisé image/décision/télémétrie dans des sessions identifiées |
| `racing/data.py`, `racing/dataset.py` | Nettoyage, fenêtres temporelles, séparation par groupes et augmentation |
| `racing/train.py` | Entraînement reproductible, validation, arrêt anticipé, sauvegardes |
| `racing/inference.py` | Chargement strict d'un checkpoint, inférence et composant DonkeyCar |
| `racing/control.py` | Direction filtrée, anticipation, accélération PI et freinage avec hystérésis |
| `racing/evaluate.py`, `racing/compare.py` | Évaluation ultérieure des prédictions et des journaux de conduite |
| `racing/plot_metrics.py` | Courbes à partir des métriques réellement produites par l'entraînement |
| `configs/race.json` | Configuration initiale centrale |
| `configs/search_space.json`, `racing/experiments.py` | Propositions d'expériences, sans lancement automatique |
| `tests/test_pipeline.py` | Tests unitaires légers et intégration sur données synthétiques |
| `docs/VERIFICATION.md`, `docs/CONNEXION_SDSANDBOX.md`, `docs/TOURS_VALIDES.md`, `docs/DATASET_BASELINE.md` | Historique des vérifications, 48 tests, tours réels et audit du dataset |
| `docs/AUDIT.md` | Diagnostic, preuves du protocole et comparaison des architectures |
| `docs/PARAMETRES.md` | Valeurs, plages proposées, justifications et méthodes de calibration |
| `docs/EVALUATION_FUTURE.md` | Procédure après entraînement, comparaison équitable et amélioration |

Les fichiers d'origine, dont `src/test_client copie.py`, le dépôt DonkeyCar et le simulateur, sont préservés. Le mode `baseline` utilise la détection visuelle des lignes et la vitesse. Le futur mode `cnn` utilise le réseau décrit ci-dessous.

## Architecture retenue

```text
Télémétrie + image RGB 255×255
             ↓
Image récente → crop vertical 30 % / 5 % → RGB 160×96 → [-1, 1]
             ↓                                  vitesse / 10
CNN : 24 → 36 → 48 → 64 → 64 canaux                   ↓
             └──────── grille spatiale 3×5 + vitesse ──┘
                                   ↓
                         denses 100 → 50 → 3
                                   ↓
                   direction maintenant / +0,25 s / +0,50 s
                         ↓                       ↓
             filtre + limite de pente     demande de virage anticipée
                         ↓                       ↓
                     steering             consigne de vitesse
                                                 ↓
                                      régulation throttle / brake
                                   ↓
                       contrôle borné → SDClient → SDSandbox
```

Le CNN apprend la direction à partir de l'image, avec la vitesse pour contextualiser les horizons futurs. Le contrôleur longitudinal utilise la plus grande direction absolue prédite pour réduire sa consigne avant un virage. Ce signal est une **demande de braquage**, pas une courbure mesurée. Les vitesses restent dans les unités de la télémétrie SDSandbox ; aucun coefficient d'adhérence ni empattement n'est supposé.

La première version utilise une image. La variante à trois images passées est implémentée (`model.frames=3`) mais doit gagner une comparaison ultérieure pour remplacer cette base. Il n'y a pas de réseau récurrent ni de modèle préentraîné à télécharger.

Les sorties utilisent `tanh` et sont bornées à [−1, +1]. Le frein est toujours séparé de l'accélérateur. Les paramètres de vitesse proposés ne sont pas des optima et ne constituent pas une garantie de tenue de piste.

## 1. Installer et vérifier les dépendances

Depuis PowerShell, dans la racine du projet :

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m racing.check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Python 3.13.4 et les versions épinglées ont passé les vérifications locales sur CPU. Python 3.12 est une alternative prévue, non testée ici. TensorFlow, Gym, torchvision et l'installation complète de DonkeyCar ne sont pas nécessaires à ce pipeline. Le `SDClient` MIT est conservé dans une copie locale identifiée par commit, car la distribution `gym-donkeycar` installée était incomplète.

L'environnement inspecté contient `torch 2.10.0+cpu` et ne détecte pas CUDA. Cela ne prouve pas l'absence physique d'un GPU. Pour un GPU NVIDIA, choisir ultérieurement une distribution CUDA de PyTorch 2.10.0 adaptée au matériel selon [les instructions officielles](https://pytorch.org/get-started/previous-versions/), puis relancer `racing.check`. Le choix `training.device=auto` utilise CUDA lorsqu'il est disponible ; l'inférence reste sur CPU par défaut pour éviter des transferts inutiles sur un petit réseau.

`racing.check` effectue seulement un passage avant synthétique et vérifie les dimensions. Il n'évalue pas la conduite.

## 2. Collecter exclusivement avec le pilote automatique

Ouvrir le simulateur manuellement, lorsque vous décidez de commencer la collecte. Le script ne démarre aucun exécutable ; il se connecte à `127.0.0.1:9091`, demande le circuit et configure la caméra.

```powershell
.\.venv\Scripts\python.exe -m racing.drive --mode baseline --connect --record --baseline-validated --group auto_06 --seconds 300
```

La conduite est entièrement automatique. `--display` ajoute un aperçu et permet un arrêt par Échap ; Espace et les flèches ne remplacent plus les commandes du pilote. Aucun exemple manuel ou CNN n'est autorisé dans l'entraînement, la validation ou le test. La collecte refuse ces modes ; leur code historique reste disponible pour compatibilité.

Réaliser **au moins cinq groupes de conduite séparés**, en changeant `--group` pour chaque nouvelle exécution indépendante. Plusieurs tours ou fragments d'une même conduite doivent garder le même groupe. Ne pas diviser artificiellement une seule séquence en cinq groupes. Des redémarrages avec les mêmes réglages ne créent pas de nouvelles conditions de conduite.

Prévoir plusieurs tours propres, des vitesses variées, les virages difficiles et des corrections depuis des positions décentrées. Une première cible de collecte est 10–20 minutes réparties en sessions, **à ajuster à la couverture réellement observée**, pas un seuil suffisant garanti. Réserver environ 60 % / 20 % / 20 % des groupes à entraînement / validation / test. Le minimum de cinq groupes laisse peu de diversité : ajouter des groupes indépendants améliore la crédibilité de la validation.

Les fichiers sont créés sous `data/sessions/<identifiant>/` :

```text
session.json       caméra, configuration, groupe, provenance
records.jsonl      une ligne par image/décision transmise
images/00000000.png ...
```

Chaque ligne conserve l'heure du simulateur, les horloges de réception/envoi, la commande décidée à partir de cette image, la vitesse, la source, la télémétrie disponible et le marqueur de récupération. `steering_angle` et `throttle` reçus restent dans `telemetry` : ils peuvent correspondre à une commande précédente. Le frein envoyé est enregistré, même si aucun écho de frein n'est exposé. Le protocole n'accuse pas l'instant exact d'application ; l'envoi réussi ne doit pas être présenté comme une mesure de cet instant.

La file d'écriture est bornée et une saturation arrête la collecte. Un changement de scène, une remise à zéro de l'heure ou une interruption impose une nouvelle session. Les sauvegardes d'images s'effectuent dans un thread séparé.

Les images historiques 160×120 restent des documents d'audit. Toute collecte utilisée ici doit avoir la caméra configurée en 255×255 RGB, FOV 150, sans fisheye. La préparation refuse une autre géométrie. Changer la caméra exige une collecte et un entraînement compatibles ; redimensionner les anciennes images ne reconstitue ni le champ de vision ni les labels manquants.

## 3. Préparer le dataset

```powershell
.\.venv\Scripts\python.exe -m racing.audit_data --sessions data/sessions --out data/audit/baseline_v1 --visual
.\.venv\Scripts\python.exe -m racing.data --sessions data/sessions --out data/prepared/baseline_v1
.\.venv\Scripts\python.exe -m racing.check --dataset data/prepared/baseline_v1
```

Cette étape produit `train.jsonl`, `val.jsonl`, `test.jsonl`, `dataset.json` et `rejected.json`. Elle vérifie la lisibilité des images, les bornes des commandes, les horodatages, la vitesse, la caméra et la qualité déclarée des labels. Un exemple invalide ou un intervalle supérieur à 150 ms coupe une séquence. Les labels futurs manquants sont masqués, jamais extrapolés.

Lire `rejected.json` et les nombres d'exemples par groupe. La seule source autorisée est **`baseline`** ; une ancienne configuration incluant `manual` est refusée à la préparation et à l'entraînement. Les images identiques entre groupes sont exclues avant de construire les labels futurs. Une collision, une perte de lignes, un |CTE| supérieur à 1, une incohérence d'horodatage ou un label non valide excluent l'exemple et coupent la séquence. Le CTE reste un indicateur conservateur, pas une mesure calibrée des limites physiques de la piste.

Les chemins d'images dans les manifests sont absolus. Après déplacement du projet, régénérer les manifests. Pour une nouvelle préparation, choisir `v2` : les répertoires existants ne sont pas écrasés. Les empreintes détectent la modification des manifests ; elles ne remplacent pas une archive immuable des images.

## 4. Lancer vous-même l'entraînement

Après une collecte suffisante et la préparation réussie :

```powershell
.\.venv\Scripts\python.exe -m racing.train --config configs/race.json --dataset data/prepared/baseline_v1 --out runs/cnn_baseline_v1 --start-training
```

Cette commande est **la première qui entraîne le réseau**. Elle n'a pas été exécutée pendant la préparation. Le drapeau `--start-training` empêche un lancement involontaire. L'entraînement ne se connecte jamais au simulateur.

Valeurs initiales : AdamW, LR `3e-4`, batch 64, 80 époques maximum, décroissance cosinus jusqu'à `1e-5`, arrêt anticipé après 12 époques sans progrès suffisant, weight decay `1e-4`, dropout 0,15, gradient clipping 1. La perte Huber (`beta=0.1`) pondère les directions actuelle / +0,25 s / +0,50 s par 0,60 / 0,25 / 0,15. Les sorties futures absentes sont exclues de la perte. Les augmentations synchronisées entre images modifient luminosité, contraste et bruit ; un retournement horizontal inverse toutes les directions et conserve la vitesse.

Le suréchantillonnage des virages utilise une correction en racine carrée, plafonnée à 4, pour limiter les déséquilibres sans effacer les lignes droites. La validation garde sa distribution naturelle. Les données de test ne sont pas utilisées pour l'arrêt anticipé.

Le dossier du run contiendra `best.pt`, `last.pt`, `metrics.csv`, les configurations, la description du dataset, les versions et les empreintes des checkpoints. Le meilleur checkpoint est choisi sur la perte de validation ; les erreurs dans les virages sont journalisées séparément. Une faible erreur hors ligne n'est pas une preuve de bons temps au tour.

Il n'y a pas de reprise automatique d'un entraînement interrompu. `last.pt` conserve aussi les états optimiseur/scheduler/scaler pour une extension future ; pour recommencer aujourd'hui, utiliser un nouveau dossier de run.

## 5. Vérifier le modèle sauvegardé, puis s'arrêter

```powershell
.\.venv\Scripts\python.exe -m racing.check --checkpoint runs/cnn_baseline_v1/best.pt --dataset data/prepared/baseline_v1
.\.venv\Scripts\python.exe -m racing.check --checkpoint runs/cnn_baseline_v1/last.pt
Get-FileHash -Algorithm SHA256 runs/cnn_baseline_v1/best.pt
.\.venv\Scripts\python.exe -m racing.plot_metrics runs/cnn_baseline_v1/metrics.csv --out runs/cnn_baseline_v1/metrics.png
```

Vérifier un nombre d'époques supérieur à zéro, une perte finie, une sortie `[1, 3]`, des poids finis et la concordance du SHA256 avec `checkpoint_hashes.json`. Le checkpoint inclut son contrat caméra/prétraitement/modèle, contrôlé au chargement avec `weights_only=True`.

**Arrêtez-vous ici et revenez avec le dossier du run.** Les commandes de pilotage CNN, d'évaluation et de calibration sont prêtes dans `docs/EVALUATION_FUTURE.md`, à utiliser dans la phase suivante.

## Limites actuelles

La connexion, les images RGB 255×255 et quatre tours du pilote déterministe sont validés dans [le rapport de conduite](docs/TOURS_VALIDES.md). Le dernier essai de 180 s utilise des décisions âgées d'au plus 56,37 ms, sous le seuil maintenu de 150 ms. Le CNN n'a pas été entraîné ni évalué. Restent à vérifier la diversité des démonstrations, la généralisation, les performances à vitesse supérieure et le règlement de compétition. Le barème officiel est absent et n'est pas inventé. Le watchdog tente un arrêt lorsqu'une connexion fonctionne encore ; une coupure totale ou un processus figé ne permet pas de garantir la réception du frein.

Les preuves et réglages sont dans [l'audit](docs/AUDIT.md), [le tableau des paramètres](docs/PARAMETRES.md) et [le protocole futur](docs/EVALUATION_FUTURE.md).
