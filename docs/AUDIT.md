# Audit technique et choix d'architecture

Audit préparatoire du 15 novembre 2026. Pas d'exécution du simulateur.

**Complément ultérieur :** la [réparation de connexion](CONNEXION_SDSANDBOX.md) a ensuite été autorisée et testée réellement. Elle corrige le framing sortant, complète `rot_x`, retire les messages non pris en charge et durcit la copie locale de SDClient. Les descriptions de copie « inchangée » ci-dessous correspondent à la préparation initiale, pas à la version corrigée.

## État initial constaté

- `DonkeySimWin/DonkeySimWin/` : application Unity compilée, ressources et DLL ; pas de sources C# locales initialement.
- `src/test_client copie.py` : `SimpleClient(SDClient)`, TCP vers `127.0.0.1:9091`, `load_scene=roboracingleague_1`, boucle de conduite de 50 secondes.
- `src/donkeycar/` : sources DonkeyCar avec pipelines TensorFlow et PyTorch génériques. Son `pyproject.toml` accepte Python ≥3.12 et <3.14. Il ne constitue pas le transport `gym_donkeycar` importé par le pilote.
- `src/camera/` : 336 fichiers PNG RGB, tous 160×120, zéro fichier corrompu et zéro doublon binaire ; aucun label associé. Résultats reproductibles dans `assets_audit.json` et `tools/audit_assets.py`.
- Aucun checkpoint propre à cette mission ni session supervisée fourni. Les données de test incluses dans les sources DonkeyCar ne sont pas des démonstrations validées du circuit.
- Python actif 3.13.4 ; PyTorch 2.10.0+cpu, NumPy 2.2.6, Pillow 11.2.1, OpenCV 4.13.0.90, Matplotlib 3.10.3. CUDA non disponible dans cette installation ; matériel GPU non établi.
- La distribution installée `gym-donkeycar 1.0.13` expose des métadonnées mais ses fichiers `core/client.py` et `core/sim_client.py` sont absents. Son utilisation directe ne serait donc pas fiable.
- Aucun `AGENTS.md` applicable trouvé dans les dossiers examinés. Les fichiers d'origine n'ont pas été modifiés.

## Défauts du pilote initial

La région utile est limitée à 50–70 % de la hauteur. La détection prend les pixels dont les trois canaux sont dans [180,255], puis calcule deux centroïdes. `tanh(error/mid)*10` est borné à ±10, alors que le protocole borne à ±1 : une grande partie des corrections sature. Le throttle vaut 0,15 lorsqu'il trouve les deux bords et zéro sinon, sans freinage. La validité des bords ne prouve pas la position sur la piste.

La boucle commande indépendamment de l'arrivée d'images, donc peut répéter une observation périmée. Les impressions à chaque image, la sauvegarde optionnelle dans le callback et les pauses de configuration peuvent retarder la réception. Les champs applicatifs sont initialisés après le démarrage du thread `SDClient`, avec une course possible lors d'une réponse très rapide. Aucune sauvegarde synchronisée des commandes n'est active.

La caméra demandée est RGB 255×255, FOV 150°, sans fisheye, seconde caméra et lidar désactivés. Un commentaire parle de niveaux de gris, mais `img_d=3` est bien envoyé. Les images historiques ont une autre résolution et leur configuration optique exacte n'est pas récupérable. Une image observée montre une route en perspective, des marquages, des structures et une ligne de départ ; aucune géométrie métrique du circuit n'en est déduite.

## Protocole : preuves et limites

Les sources publiques de référence sont archivées dans `upstream/manifest.json` avec commits exacts. Une lecture des instructions IL des **DLL fournies** a ensuite été effectuée avec dnfile/dncil ; elle ne charge ni n'exécute le code Unity. Preuves dans `binary_protocol_il.txt`, empreinte principale `5dee00e9695e0c453373c29d87592cf445a618b789ecc07806550acc28043eeb`.

La version `v25.10.06` est celle indiquée dans la mission ; l'empreinte identifie le binaire inspecté sans prétendre établir sa provenance de publication.

| Élément | Constat vérifié statiquement | Limite |
|---|---|---|
| Format contrôle | JSON `control`, valeurs numériques sérialisées comme chaînes ; `float.Parse(...InvariantCulture)` | Pas d'accusé d'application par commande |
| Steering | `TcpCarHandler.OnControlsRecv` borne à [−1,+1], multiplie par `steer_to_angle=16` et appelle `RequestSteering` | Dans `Car`, bornage supplémentaire par `maxSteer`, valeur de scène susceptible d'être sérialisée |
| Signe | Le signe positif est transmis sans inversion à `WheelCollider.steerAngle` ; convention attendue : droite positive | Confirmation visuelle sur le véhicule actif non effectuée |
| Throttle | Protocole [−1,+1] | Le nouveau pilote restreint à [0,1], sans marche arrière |
| Brake | Protocole [0,1], `RequestFootBrake` après `RequestThrottle` | Efficacité dépend du composant de véhicule réellement instancié |
| Frein `Car` | `requestBrake * maxBreakTorque` transmis à `brakeTorque` des roues | Couple de frein sérialisé et distance de freinage non établis |
| Frein `UnityStandardCarAdapter` | Transmet `footBrake` positif à `CarController.Move`, qui borne son paramètre à [−1,0] puis le nie | Dans ce chemin, le frein réseau positif peut être annulé. Ne pas remplacer par un frein négatif : le protocole le borne à zéro |
| Vitesse | `magnitude(GetVelocity()/8)` | Scalaire non signé ; ne pas utiliser la phrase du README Gym « vitesse négative en marche arrière » pour ce binaire |
| Accélération | `GetAccel()/8`, canaux x,y,z | `Car` utilise une différence de vitesse locale / deltaTime ; l'adaptateur a une autre définition. Ce n'est pas automatiquement une mesure inertielle exploitable d'adhérence |
| Gyroscope | `GetGyro`, x,y,z ; pose pitch/yaw/roll | Repère et calibration à confirmer avant exploitation dynamique |
| Position, vitesse vectorielle, CTE | Conditionnés par `extendedTelemetry` | Leur présence en compétition n'est pas garantie ; ils ne pilotent pas le CNN |
| Progression | `activeNode`, `totalNodes` si PathManager disponible | Ne constituent pas seuls une validation officielle de tour |
| Tours | Événement `collision_with_starting_line`, index et `timeStamp` | Une recroisée de ligne ou un trajet inversé peut donner un faux tour |

`speed`, `time`, `image`, `steering_angle`, `throttle`, `hit`, accélérations, gyroscopes et orientations apparaissent dans le chemin de télémétrie inspecté. Aucune capture de messages du simulateur n'a été effectuée : les champs réellement reçus seront vérifiés à la collecte. Le pipeline exige `image`, `speed` et `time`, sans les remplacer silencieusement par des zéros. Les données facultatives sont conservées lorsqu'elles existent.

Le facteur 8 impose de raisonner dans les unités transmises. Aucune adhérence maximale, longueur d'empattement, largeur utile de piste, inertie, saturation du moteur ou décélération effective n'a été mesurée. La formule `sqrt(a_lat_max / abs(curvature))` n'est donc pas activée. La direction normalisée n'est pas une courbure ; même `tan(delta)/L` demanderait un modèle et un empattement calibrés, avec prudence en glissement.

## Communication réutilisée

`SDClient` construit une socket TCP, lance son thread de lecture, sépare les messages par fin de ligne, parse le JSON et appelle `on_msg_recv`. Il offre `send`, `send_now`, `stop`, `aborted` et `poll_socket_sleep_sec`.

`send` n'est qu'un emplacement de message remplaçable, pas une file fiable. `send_now` appelle `sendall`; la socket est non bloquante dans la boucle de réception. Une erreur d'envoi est traitée comme une anomalie, sans supposer la livraison. `stop` contient un `join` sans délai et le traitement de fin de socket ne suffit pas à garantir un arrêt de l'application.

Une copie MIT inchangée du client et de son utilitaire est incluse dans `racing/vendor` afin de préserver cette communication sans installer les dépendances Gym ni réécrire la boucle TCP. `RaceClient` ajoute :

- Initialisation avant le thread, stockage de la dernière observation seulement, conservation séparée des événements.
- Envois courts sérialisés par verrou via `send_now`, aucun appel au slot `send`.
- Configuration hors du callback et vérification de la résolution réellement reçue.
- Surveillance indépendante de la réception et de la production de commandes ; défaut verrouillé empêchant une prédiction retardée de réaccélérer.
- Arrêt borné de la connexion ; tentative d'envoi d'un frein avant fermeture.

Le contrôle vise 20 Hz, le polling socket 5 ms. Ce sont des réglages initiaux, pas des fréquences mesurées. L'ancien `sleep(0.01)` ne démontrait pas un vrai contrôle à 100 Hz. Le journal mesure la durée prétraitement+inférence+transfert de résultat et le délai réception→envoi. L'âge capture→réception reste inconnu : l'heure du simulateur n'est pas synchronisée avec l'horloge monotone du client. Un backlog TCP peut donc nécessiter une calibration ultérieure de l'âge réel des images.

## Choix du réseau

| Option | Atouts | Coûts / risques | Choix |
|---|---|---|---|
| PilotNet classique | Référence de Behavioral Cloning compacte, cinq convolutions et régression | Prétraitement et résolution d'origine différents ; pas d'anticipation explicite de vitesse | Inspiration, pas copie supposée optimale |
| CNN personnalisé monovue | Faible coût, contrôle précis des dimensions, grille conservant la position latérale | Nécessite des démonstrations variées ; pas d'état temporel implicite | Base retenue |
| CNN multi-sorties steering/throttle/brake | Imite les décisions complètes d'un expert | Apprend aussi son allure lente et sa coordination ; calibration plus difficile avec peu de données | Commandes longitudinales enregistrées mais non directement prédites en V1 |
| CNN multi-horizons de direction | Anticipation des virages avec les mêmes séquences étiquetées | Futurs conditionnés par l'allure et la trajectoire de démonstration | Trois sorties retenues, vitesse normalisée en entrée |
| CNN à trois images | Peut désambiguïser déplacement et oscillation, faible ajout de paramètres | Historique et timing stricts, démarrage et trous de séquence ; gain non établi | Variante disponible, pas activée par défaut |
| Réseau récurrent / modèle profond | Plus de capacité temporelle | Complexité, état et latence, risque de surapprentissage sans données | Différé |

Les convolutions ont des kernels 5/5/5/3/3, strides 2/2/2/1/1, padding 2/2/2/1/1 et activations ELU. À 160×96, la grille finale 64×12×20 est moyennée spatialement par blocs 4×4 vers 64×3×5, concaténée avec la vitesse, puis passée dans 100 et 50 neurones, dropout 0,15 et trois sorties `tanh`. Pas de BatchNorm pour éviter des statistiques de petits lots délicates. Le pooling est fixe et compatible avec l'option déterministe sur GPU ; les tailles configurables doivent être des multiples de 40 en largeur et 24 en hauteur.

La résolution brute 255×255 reprend la caméra du script. La découpe conserve 65 % de sa hauteur puis redimensionne vers 160×96 avec la même interpolation bilinéaire partout. Ce redimensionnement est une transformation fixe ; il ne reconstitue pas une calibration géométrique. Contrairement au masque 50–70 %, il conserve une zone d'anticipation plus large. FOV et crop demeurent à comparer après les premiers résultats.

Le choix est justifié par la petite quantité actuelle de données et l'environnement CPU. Aucune comparaison expérimentale des architectures n'a été menée. Le meilleur modèle en compétition devra être déterminé en boucle fermée, à conditions égales.

## Sources primaires

- [SDSandbox, handler TCP](https://github.com/tawnkramer/sdsandbox/blob/ec04c400f045222375ec82e649bb051fb17aad41/sdsim/Assets/Scripts/tcp/TcpCarHandler.cs) : protocole et télémétrie, confrontés au DLL local.
- [SDSandbox, Car](https://github.com/tawnkramer/sdsandbox/blob/ec04c400f045222375ec82e649bb051fb17aad41/sdsim/Assets/Scripts/Car.cs) : actionneurs et dynamique logicielle.
- [SDSandbox, adaptateur Unity](https://github.com/tawnkramer/sdsandbox/blob/ec04c400f045222375ec82e649bb051fb17aad41/sdsim/Assets/Scripts/UnityStandardCarAdapter.cs) : chemin alternatif du frein.
- [gym-donkeycar, SDClient](https://github.com/tawnkramer/gym-donkeycar/blob/a1f4ca6961e17a6929c1f2e883358a1edafb479d/gym_donkeycar/core/client.py) : transport conservé, licence MIT jointe.
- [Bojarski et al., End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) : référence PilotNet et apprentissage image→direction.
- [PyTorch, installation](https://pytorch.org/get-started/locally/) : compatibilité et choix des distributions CPU/CUDA.

Ces sources étayent l'audit et les principes d'architecture ; elles ne prouvent aucune performance de ce nouveau pilote sur ce circuit.
