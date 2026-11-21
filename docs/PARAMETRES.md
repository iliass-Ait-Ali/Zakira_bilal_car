# Paramètres initiaux et optimisation prévue

Toutes les valeurs modifiables du pilote sont dans `configs/race.json`. Les propositions par étapes sont dans `configs/search_space.json`. **Aucune valeur n'a été optimisée par entraînement ou conduite.**

Depuis cet inventaire initial, la [réparation de connexion](CONNEXION_SDSANDBOX.md) a validé le protocole. Une mission suivante a réglé le pilote déterministe et [validé quatre tours](TOURS_VALIDES.md) : ses paramètres retenus sont dans `configs/race.json`, rubrique `baseline`. Les paramètres CNN ci-dessous restent proposés, sans entraînement ni optimisation du réseau.

Statuts : **V** = vérifié statiquement dans les fichiers / DLL, **I** = choix initial d'ingénierie, **C** = calibration ou validation expérimentale nécessaire. Aucun paramètre n'a le statut « validé par essai dans le simulateur ».

Les unités de vitesse ci-dessous sont celles du champ `speed` du simulateur, pas une conversion affirmée en km/h. Les plages proposées délimitent des essais progressifs, pas des plages de sécurité démontrées.

| Paramètre / rôle | Valeur initiale | Plage autorisée ou recherche proposée | Statut, justification et influence | Optimisation ultérieure |
|---|---:|---|---|---|
| Commande steering | [−1,+1] | Borne du protocole | V : sorties réseau et commandes cohérentes ; évite la saturation ×10 historique | Ne pas modifier la borne réseau |
| Conversion steering interne | ×16 | Constante du handler ; autre borne `maxSteer` possible | V : angle demandé en degrés ; ne prouve pas l'angle effectif sous glissement | Vérifier le prefab et la réponse visuelle |
| Signe steering | Positif à droite attendu | ±1 selon convention vérifiée | C : signe conservé jusqu'aux roues ; pas de test effectué | Courte commande manuelle à faible allure |
| Throttle protocole | [−1,+1] | Pilote restreint à [0,1] | V/I : interdit la marche arrière dans ce pipeline | Ne pas confondre marche arrière et freinage |
| Brake protocole | [0,1] | Amplitude de course ≤0,7 initialement | V/C : signal correct ; efficacité liée au composant actif | Vérifier décélération, zéro moteur et arrêt |
| `camera.img_w/img_h/img_d` | 255/255/3 | Fixés par dataset ; alternative 160×120 dans une collecte séparée | V/I : reprend la demande du script, vérifie la taille reçue | Comparer coût réseau et visibilité après nouvelle collecte |
| `camera.img_enc` | PNG | PNG ; JPG exige contrat et collecte cohérents | I : sans pertes, coût réseau plus élevé | Mesurer latence avant de changer |
| `camera.fov` | 150° | 90–150°, collecte distincte | V/C : garde la vue demandée, déformation grand-angle possible | Comparaison contrôlée du champ de vision |
| Fisheye / offsets | 0 / 0 | Conserver initialement | V/C : zéro peut conserver une valeur par défaut côté simulateur | Documenter le montage effectivement reçu, ne pas présumer une position métrique |
| `preprocess.width/height` | 160×96 | 120×72, 160×96, 200×120 ; multiples 40/24 | I : compromis détail/coût et grille spatiale 3×5 | Comparer erreur de virage puis réussite/latence |
| `crop_top/crop_bottom` | 0,30 / 0,05 | Haut 0,20–0,40 ; bas 0–0,10 | I : conserve davantage d'anticipation que l'ancien ROI | Vérifier visuellement horizon et proximité ; refaire préparation et entraînement |
| RGB / normalisation | `/127.5 - 1` | Fixe, [−1,1] | I : même fonction pour collecte préparée et inférence | Ne pas changer séparément en production |
| `model.frames` | 1 | 1 ou 3 | I : base rapide ; trois images peut stabiliser le mouvement | Ablation sur mêmes groupes, comparaison à allure égale |
| `frame_spacing_s` | 0,05 s | 0,05–0,10 s pour 3 images | I/C : cohérent avec un contrôle visé à 20 Hz | Mesurer intervalles et rejets temporels avant modification |
| `horizons_s` | 0 / 0,25 / 0,50 s | Futur proche 0,15–0,35 ; lointain 0,4–0,8 s | I : anticipation sans énorme ambiguïté des trajectoires futures | Comparer freinage tardif, MAE par horizon et tours |
| Convolutions | 24/36/48/64/64 | Canaux ×0,75 à ×1,5 en variante codée | I : 232 751 paramètres en configuration de base | Augmenter seulement si sous-apprentissage établi |
| Kernels / stride / padding | 5,5,5,3,3 / 2,2,2,1,1 / 2,2,2,1,1 | Architecture fixe dans `model.py` | I : réduction rapide, conservation des bords | Modification d'architecture = nouveau modèle et tests de forme |
| Activation / grille | ELU / 3×5 | ELU initialement ; pas de pooling global | I : garde position horizontale et hauteur | Ablation ciblée seulement si besoin |
| `model.dense` | 100 / 50 | [64,32] à [128,64] | I : fusion vitesse et vision, coût modéré | Courbes train/validation et latence |
| `model.dropout` | 0,15 | 0,10–0,25, éventuellement 0,30 | I : limite l'overfitting | Ajuster selon écart train/validation |
| `speed_scale` | 10 | ≥ vitesse maximale admise, fixer avant collecte | I/C : division de l'entrée ; dépassement refusé | Étendre seulement avec démonstrations au nouveau régime |
| Perte / `huber_beta` | Huber / 0,1 | 0,05–0,20 | I : moins sensible aux labels aberrants que MSE | Comparer virages et erreurs extrêmes |
| `loss_weights` | 0,60/0,25/0,15 | Actuel 0,5–0,8 ; solde aux futurs | I : privilégie trajectoire immédiate, anticipe l'allure | Augmenter futur seulement si labels fiables |
| Optimiseur | AdamW | Fixe initialement | I : réglage stable, découplage weight decay | Ne pas multiplier les optimiseurs sans signal |
| LR initial / minimum | 3e-4 / 1e-5 | Initial 1e-4–6e-4 | I : première priorité hors ligne | Deux variantes, mêmes groupes et seed |
| Scheduler | Cosinus sur 80 époques | Suivre le budget d'époques retenu | I : réduction progressive | Lire les courbes avant allongement |
| `batch_size` | 64 | 32–128 selon mémoire | I/C : stable sur petit CNN ; CPU possible | Réduire sur manque mémoire, puis recontrôler LR |
| `epochs` | 80 maximum | 40–120 | I : plafond, pas obligation d'atteindre 80 | Courbes et arrêt anticipé |
| `patience/min_delta` | 12 / 1e-4 | Patience 8–20 | I : évite de poursuivre sans progrès | Ajuster au bruit réel de validation |
| `weight_decay` | 1e-4 | 3e-5–3e-4 | I : régularisation des poids | Avec dropout, après LR |
| `clip_grad_norm` | 1,0 | 0,5–5 | I : protège contre pics de gradients | Contrôler seulement si instabilités constatées |
| `seed/deterministic` | 42 / vrai | Seeds 42,43,44 pour confirmation | I : reproductibilité, pas garantie bit-identique entre matériels | Répéter seulement les finalistes |
| `device/amp/threads/workers` | auto / vrai CUDA / 4 / 0 | CPU/CUDA ; threads 1–8 ; workers 0–4 | I/C : Windows compatible, AMP désactivé sur CPU | Mesurer ultérieurement, réduire concurrence avec Unity |
| `val_fraction/test_fraction` | 0,2 / 0,2 | 0,15–0,25 par groupes | I : pas de mélange d'images voisines entre splits | Garder test gelé après création |
| `min_groups` | 5 | 5 minimum, davantage préférable | I : trois partitions non vides, diversité minimale | Ajouter sessions indépendantes, pas fragments artificiels |
| `max_gap_s` | 0,15 s | 0,10–0,25 s | I : coupe des trajectoires discontinues | Examiner intervalles et pertes de trames |
| `time_tolerance_s` | 0,06 s | 0,025–0,075 s | I : choix du voisin temporel ; historique distinct exigé | Resserrer si acquisition régulière ; vérifier les masques |
| `max_decision_age_s` | 0,15 s | 0,08–0,20 s | I : refuse des labels trop tardifs | Mesurer réception→envoi, ne pas allonger pour masquer un blocage |
| `balance_power/max_sample_weight` | 0,5 / 4 | Puissance 0,3–0,7 ; plafond 2–6 | I : rééquilibre sans supprimer les lignes droites | Histogrammes et erreur selon angle |
| `steering_bins` | ±0,05 / ±0,2 / ±0,6 | Frontières symétriques à adapter | I : distingue ligne droite, léger virage, virage fort | Vérifier couverture par bin |
| `recovery_weight` | 1,5 | 1–2 | I : garde les corrections difficiles | Vérifier leur qualité avant surpondération |
| Flip horizontal | p=0,5 | 0–0,5 | I : inverse tous les steering ; vitesse inchangée | Désactiver si indices visuels asymétriques nuisent |
| Brightness / contraste | ±15 % / ±15 % | ±5–20 % | I : variation photométrique légère | Inspecter exemples, éviter de détruire les marquages |
| Bruit normalisé | σ=0,01 | 0–0,02 | I : faible bruit, même prétraitement de base | Ablation si images trop dégradées |
| `frequency_hz/poll_s` | 20 Hz / 5 ms | Contrôle 15–30 Hz ; polling 1–10 ms | I/C : coût raisonnable sur Windows | p95 latence, fréquence reçue et trames sautées |
| `steering_rate` | 4 unités/s | 2–8 unités/s | I : max 0,2 par 50 ms, réduit oscillation | Augmenter si retard au virage ; réduire si oscillations |
| `steering_tau_s` | 0,04 s | 0–0,08 s | I : filtre exponentiel avec dt réel | Garder au minimum nécessaire pour lisser |
| `speed_straight` | 5 | 3–8 initialement, au plus `speed_scale` | I/C : consigne de ligne droite, aucune réussite démontrée | Monter par 10–15 % après tours propres répétés |
| `speed_turn` | 2 | 1,5–3,5 | I/C : plancher de consigne virage, pas vitesse garantie sûre | D'abord fiabilité dans les virages difficiles |
| `turn_gain` | 2,5 | 1,5–4 | I/C : sensibilité de `v_straight/(1+gain*demande)` | Réduire si sous-vitesse ; augmenter si arrivée trop rapide |
| `speed_rise_per_s` | 2 | 1–4 | I : réaccélération de la consigne, baisse immédiate | Sorties de virage et taux de sortie de piste |
| `throttle_kp/ki` | 0,25 / 0,08 | Kp 0,10–0,50 ; Ki 0–0,15 | I/C : vitesse corrigée, intégrale limitée | Identification sur lignes droites, ajuster Kp puis Ki |
| `integral_limit` | 1,5 | 0,5–3 | I : anti-emballement de l'intégrale ; gel à saturation | Vérifier dépassement après saturation |
| `throttle_ff/max` | 0,08 / 0,60 | FF 0–0,15 ; max 0,3–1 | I/C : compensation initiale et réserve d'accélération | Augmenter plafond avec marge de contrôle validée |
| `brake_kp/max` | 0,30 / 0,70 | Kp 0,10–0,60 ; max 0,3–1 | I/C : freine sur survitesse, aucun throttle simultané | Mesurer la décélération avant réglage de course |
| Hystérésis frein entrée/sortie | 0,35 / 0,10 | Entrée 0,2–0,6 ; sortie < entrée | I : évite les microfreinages alternés | Examiner cycles frein/roue libre |
| `emergency_brake` | 1 | [0,1] | I/C : tentative d'arrêt sur anomalie | Confirmer le chemin physique de freinage |
| `timeout_s/max_frame_age_s` | 0,4 / 0,15 s | Timeout 0,2–0,6 ; âge 0,08–0,20 | I : verrouille l'arrêt sur données/commandes anciennes | Mesurer les queues ; conserver marge sans accepter longues pertes |
| `startup_timeout_s` | 20 s | 10–60 s | I : borne chargement, avant armement | Adapter au temps de chargement de la machine |
| `manual_throttle` | 0,20 | 0,05–0,50 progressivement | I/C : collecte variée, pas expert automatique | Modifier entre groupes après vérifications de commande |
| Manuel braquage/recentrage | 2 / 3 unités/s | 1–4 / 1–5 | I : contrôle clavier progressif | Adapter à la qualité des démonstrations |
| Référence throttle/gain | 0,15 / 10 | Garder pour comparaison historique, puis variante identifiée | V/I : reproduit la saturation effective, freine à perte de bords | Ne l'utiliser comme expert qu'après validation |
| `evaluation.cte_limit` | null | Aucun seuil inventé | C : largeur piste et unités CTE inconnues ; champ parfois absent | Annoter sorties ou calibrer seuil, rapporter couverture |
| `minimum_lap_s` | 5 s | À adapter au circuit | I : simple filtre anti-rebond ; pas homologation de tour | Vérifier événements et ordre des checkpoints |
| Score officiel | null | Barème non fourni | C : aucun score artificiel | Ajouter un plugin de calcul après réception du règlement |

La recherche est séquentielle : qualité des labels et splits → LR → régularisation → temporalité → vitesse → lissage. Ne pas utiliser le test final pour choisir ces réglages. Toute modification du modèle, de la caméra ou du prétraitement change le contrat du checkpoint ; régénérer un dataset préparé avec la nouvelle configuration, même si les images sources restent réutilisables.
