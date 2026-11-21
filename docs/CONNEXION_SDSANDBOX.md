# Correction et validation réelle de la connexion SDSandbox

Intervention du 15 novembre 2026, après autorisation explicite de tests techniques et d'un très court essai baseline. **Aucun entraînement, fine-tuning, enregistrement de dataset ou tour complet n'a été lancé.**

## Cause établie

Le port `127.0.0.1:9091` appartient au PID **9956**, processus `donkey_sim`, dont le chemin observé est bien `DonkeySimWin/DonkeySimWin/donkey_sim.exe` dans ce projet. Le listener n'était pas un autre service.

Trois anciens runs contenaient `car_loaded` puis le timeout, sans compter ni expliquer les images rejetées. La télémétrie n'était donc pas nécessairement absente : la lecture passive du serveur a effectivement reçu des images et de la télémétrie. Le blocage venait de la configuration non appliquée et d'un diagnostic insuffisant.

### 1. Le client ajoutait un saut de ligne interdit en sortie

`RaceClient.send_packet` ajoutait `\n` à chaque JSON. Le DLL local utilise un parseur asymétrique :

- **Unity → Python** : objets JSON terminés par `\n`.
- **Python → Unity** : objets concaténables `{...}{...}`, dont le dernier caractère doit être exactement `}` ; le parseur ne supprime pas le saut de ligne.

La méthode `JsonTcpClient.ExtractJsonFromStream` sépare sur `{`, puis teste le dernier caractère de chaque fragment. Les commandes envoyées avec `\n` échouaient à ce test. Le journal Unity contient réellement :

```text
Unexpected partial JSON object in the middle of the TCP buffer !
```

suivi des messages `load_scene`, `car_config`, `cam_config` et `control` ajoutés par le client. C'est une **cause observée dans le serveur**, confirmée par l'inspection statique du DLL. Les commandes sortantes sont maintenant compactes et sans terminateur ; la réception conserve le traitement par lignes.

### 2. La caméra nécessitait un champ absent

`TcpCarHandler.ParseCamConfig` lit obligatoirement `rot_x`. L'ancien script fourni et le nouveau client omettaient tous deux ce champ. Ce défaut aurait empêché la configuration même après correction du framing. Le transport ajoute désormais `rot_x="0"` : dans `CameraSensor.SetConfig`, des rotations toutes nulles préservent l'orientation du prefab.

La résolution par défaut de 160×120 pouvait être reçue et rejetée silencieusement par `drive.py`. Les images incompatibles sont désormais comptées et expliquées. Une image transitoire de cette taille a réellement été observée, puis les images 255×255 sont arrivées après application de la configuration.

Les messages `cam_config_b` et `lidar_config` avec seulement `enabled=false` sont aussi incompatibles avec ce binaire : leurs parseurs attendent les paramètres complets et ne consultent pas ce drapeau. Ils ne sont plus envoyés. Les capteurs optionnels restent à leurs valeurs par défaut ; les paquets observés n'incluaient ni `imageb` ni lidar. `racer_info` est également ignoré par le dispatcher actif, ce que `Player.log` a confirmé pendant le premier essai ; ce message a été retiré de la configuration finale. Le nom de la voiture reste transmis par `car_config`.

### 3. L'erreur `file descriptor ... (-1)` venait de l'arrêt

L'ancien `close()` fermait la socket avant de joindre le thread. Celui-ci pouvait être entre son test de boucle et `select()`, puis utiliser le descripteur devenu `-1`. Cette course est distincte du timeout caméra et survient lors de son nettoyage.

L'arrêt suit désormais cet ordre : désarmement du watchdog, tentative explicite de freinage lorsque possible, signal d'arrêt réseau, `shutdown()` pour réveiller la réception, attente bornée du thread, puis fermeture du descripteur. Une EOF réelle déclenche une erreur de déconnexion explicite ; elle ne provoque plus une boucle de réception vide. Les écritures partielles non bloquantes sont complétées jusqu'à une échéance, et aucun envoi n'est accepté après fermeture.

## Séquence actuelle

`SimulatorSession` pilote le démarrage indépendamment de la stratégie manuelle, baseline ou CNN.

Depuis le menu :

```text
Connexion TCP
→ scene_selection_ready reçu
→ get_scene_names envoyé, liste vérifiée
→ load_scene roboracingleague_1 envoyé
→ car_loaded reçu
→ car_config envoyé
→ délai de configuration 0,2 s, sans bloquer la réception
→ cam_config complet envoyé
→ première image conforme décodée
→ READY puis DRIVE pour un mode de conduite
```

Depuis une piste déjà ouverte, `car_loaded` peut être le tout premier événement. Le client retourne alors au menu avec `exit_scene`. Ce binaire n'envoie pas toujours `scene_selection_ready` lors d'un retour au menu ; une réponse à `get_scene_names` confirme donc l'installation du handler de menu. Le circuit demandé est ensuite vérifié dans la liste et chargé. Cela évite d'envoyer `load_scene` à un handler de voiture qui ne le connaît pas, ou de conduire silencieusement sur la mauvaise piste.

Le délai maximal de démarrage reste **20 secondes**, inchangé. Le contrôle n'est armé qu'après réception d'une image conforme. Les événements dupliqués pendant la configuration ne provoquent pas de rechargement. Un rechargement après activation reste une erreur explicite.

## Fichiers concernés

| Fichier | Changement |
|---|---|
| `racing/transport.py` | JSON sortant correct, caméra complète, commandes compatibles, compteurs et arrêt propre |
| `racing/vendor/client.py` | EOF, buffer de bytes, messages fragmentés/coalescés, écritures partielles et cycle de vie de socket |
| `racing/vendor/util.py` | Décimales à virgule de Unity corrigées sans altérer les chaînes/base64 ou tableaux JSON |
| `racing/session.py` | États menu/scène/voiture/caméra, erreurs de décodage détaillées et maintien de la dernière observation jusqu'à l'échéance de contrôle |
| `racing/diagnostics.py` | Étapes horodatées, compteurs, rejets, erreurs et journaux JSON sans base64 |
| `racing/diagnose.py` | Test limité de réception d'images, sans accélération ni dataset |
| `racing/drive.py` | Démarrage partagé, `--debug`, `--diagnose`, `--frames`, `--image-out`, sauvegarde des diagnostics même en cas d'échec |
| `tests/test_connection.py` | 15 tests de non-régression sur le protocole et les sockets |
| `tests/test_pipeline.py` | Adaptation du test d'envoi aux messages compacts et à l'erreur de connexion explicite |

Le CNN, l'entraînement, les données sources, le prétraitement et `configs/race.json` n'ont pas été modifiés pendant cette réparation. `rot_x=0` est un défaut de sérialisation du protocole, sans changement des paramètres de caméra stockés ni de leur hash. Le contrat reste **255×255, RGB, PNG, FOV demandé 150°**, puis prétraitement 160×96. Le simulateur ne renvoie pas d'accusé du FOV : le paramètre envoyé et sa lecture dans le DLL sont vérifiés ; aucune mesure optique indépendante du FOV n'a été faite.

## Tests réellement exécutés

1. Suite initiale : **20 tests réussis**, avant toute connexion de diagnostic.
2. Inspection du processus propriétaire du port, des anciens runs et de `Player.log`.
3. Lecture statique du DLL, sans exécution de code .NET par l'outil d'audit.
4. Probes passives courtes, sans commandes d'accélération, confirmant la présence de télémétrie et les décimales à virgule de l'installation française.
5. Suite finale : **35 tests réussis** ; compilation Python sans erreur. Les tests TCP utilisent un serveur local synthétique sur un port éphémère, pas SDSandbox.
6. Tests réels limités ci-dessous, avec le processus Unity déjà ouvert par l'utilisateur.

| Test réel | Résultat observé |
|---|---|
| Diagnostic depuis une piste déjà chargée | Circuit demandé rechargé ; 7 télémétries, 3 images RGB 255×255 décodées, une image transitoire 160×120 rejetée ; aucune accélération |
| Baseline avec affichage, 5 secondes | 107 télémétries, 101 images décodées, 50 décisions avec throttle positif ; 49 échos de throttle positif ; déplacement constaté |
| Baseline finale avec affichage, 3 secondes | 66 télémétries, 62 images décodées, 47 décisions ; 46 échos de throttle positif ; déplacement constaté ; arrêt sans faute |
| Diagnostic final depuis le menu | `scene_selection_ready` réellement reçu, circuit chargé ; 7 télémétries, 3 images RGB 255×255 décodées ; aucune accélération |

Tous ces runs ont **zéro erreur réseau et zéro erreur de parsing JSON côté client**. Les deux essais baseline ont un `end.json` avec `fault: null`. Aucune exception de descripteur négatif n'est apparue. Le dernier contrôle du port ne montre plus de connexion cliente établie, uniquement le listener du simulateur.

Le test baseline de 3 secondes couvre environ 3 secondes du temps simulateur. Les positions reçues passent de `[2.983010, 0.069982, -3.244590]` à `[1.584823, 0.070224, -3.221384]`. Ce sont des preuves de mouvement, **pas des résultats de performance en course**. La cadence est une cible de 20 Hz ; ces courts essais avec affichage ne démontrent pas une cadence exacte sur cette machine.

Les données de preuve compactes sont dans `connection_evidence.json`. Les traces complètes sont conservées sous :

```text
runs/connection_checks/diagnose_ee49eca5ee0b49f8a1c63907d45a015e/
runs/connection_checks/61edbf5e423b43fdb1d6e84b9c7940c7/
runs/connection_checks/4a698e25423144fc8ee94efb41b03af7/
runs/connection_checks/diagnose_2e68ff73e7ba4691a3280a10d05b8ab6/
runs/connection_checks/first_camera.png
```

Une seule image a été sauvegardée à titre de diagnostic ; aucun dossier de collecte supervisée n'a été créé. L'image est visible, en 255×255, avec la piste et le capot du véhicule. Les instructions IL pertinentes sont dans `connection_protocol_il.txt` et `connection_scene_il.txt`. Un extrait de télémétrie réelle, avec l'image masquée, est dans `connection_raw_telemetry_redacted.txt`.

## Commandes pour reproduire

Depuis la racine du projet, avec le simulateur ouvert et à l'écoute sur 9091.

Diagnostic de connexion, chargement et première image, sans accélérer :

```cmd
.\.venv\Scripts\python.exe -m racing.drive --diagnose --connect --debug
```

Vérifier dix images conformes et sauvegarder une seule image témoin :

```cmd
.\.venv\Scripts\python.exe -m racing.drive --diagnose --connect --frames 10 --image-out runs/diagnostics/camera.png
```

Tester 30 secondes de conduite déterministe avec l'interface existante :

```cmd
.\.venv\Scripts\python.exe -m racing.drive --mode baseline --connect --display --seconds 30
```

Ajouter `--debug` à cette commande pour afficher les échanges de configuration. La durée de conduite commence maintenant après `DRIVE`, et non pendant le chargement. La validation réalisée ici s'est limitée à 5 puis 3 secondes ; la commande de 30 secondes n'a pas été exécutée pendant cette intervention.

Tests de non-régression sans simulateur :

```cmd
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Chaque exécution affiche le chemin exact de ses traces. `protocol_summary.json` donne l'état atteint, les types et nombres de messages, les télémétries, tailles décodées, rejets et raisons, erreurs réseau, commandes et temps de première occurrence. `protocol.jsonl` donne les étapes et leurs délais. Même si l'exécution échoue, ces deux fichiers sont sauvegardés. Un diagnostic réussi ne masque pas les images transitoires rejetées.

## Limites

La connexion et la réception caméra sont validées réellement depuis le menu et une piste déjà ouverte. Le pilotage baseline démarre et déplace réellement la voiture. Sa capacité à terminer le circuit, la fiabilité sur de nombreux tours, ses réglages, le freinage maximal et les performances du futur CNN ne sont pas évalués par ces tests courts. Le signe, les sorties de piste et les temps au tour restent à étudier dans la phase dédiée. Aucune reconnection automatique en course n'est tentée après une vraie perte réseau : le défaut est affiché et les commandes s'arrêtent.
