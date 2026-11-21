# Journal : travail fait en autonomie (20 novembre 2026)

Tu m'as demandé de continuer à améliorer le projet et de prendre les décisions pendant ton absence. Ce journal note chaque décision et sa raison.

## Point de départ (avant ton départ)

| Modèle | Circuit principal | warren | mini_monaco / generated_road |
|---|---|---|---|
| CNN v2 (circuit principal seulement) | 42,1 s par tour (réglage b) | 1 tour sans collision, jamais vu à l'entraînement | roule sans collision, jamais vu |
| CNN v3 (principal + warren v1) | 41,6–42,2 s, 3 tours validés | 3,4 tours sans collision en 3 min, mais 1 seul validé | roule sans collision, jamais vu |

## Décisions

1. **Le professeur de warren braquait trop peu.** Sa limite de braquage était fixée à 0,4 pour éviter de sortir au démarrage. Le CNN a appris cette limite et sortait un peu dans le grand virage à droite.
   Décision : la limite de 0,4 ne s'applique plus que pendant les 3 premières secondes, puis elle passe à 0,7 (`startup_s`, `startup_max_steering` dans [racing/baseline.py](../racing/baseline.py)).
   Résultat : l'écart maximal d'un tour du professeur passe d'environ 1,6 à **0,11**.
2. **Nouvelles données warren.** Les 4 anciennes sessions warren sont mises de côté dans `data/excluded_sessions/warren_v1` (pas supprimées). J'ai enregistré 5 nouvelles sessions (`warren_b1` à `warren_b5`) avec le meilleur professeur.
3. **Professeur « ligne jaune ».** mini_monaco et generated_road n'ont pas de lignes blanches nettes, mais elles ont une ligne jaune au centre. J'ai ajouté un mode `guide: "yellow"`, activé seulement dans les configs de ces circuits.
   Vérification hors simulateur : sur toutes les photos, il braque dans le même sens que le CNN. Sur generated_road, la ligne de référence du simulateur est décalée de −2,2, comme sur warren (−2,1).
4. **mountain_track est gardé comme circuit de contrôle.** Il ne sert jamais à l'entraînement : c'est sur lui qu'on vérifie si le modèle sait conduire sur un circuit inconnu.

5. **warren, nouvelles données : 28 tours validés sur 28.** Cinq sessions de 5 minutes, soit environ 29 000 images valides, sans aucune coupure.
6. **Mise au point du professeur « ligne jaune »** (essais dans le simulateur) :
   - La ligne de generated_road est **pointillée**. Dans un trou, le professeur freinait et s'arrêtait. Corrections : il regarde plus loin dans l'image, et il peut rouler jusqu'à 0,8 s sans ligne sans accélérer (`lost_grace_s`). Ces images-là ne servent jamais à l'apprentissage.
   - La ligne de generated_road est d'un jaune délavé, d'où l'option `yellow_pale`.
   - Sur mini_monaco, la route beige donnait de faux points jaunes. À l'ombre, la vraie ligne devenait terne. Correction : on compare chaque pixel au reste de sa ligne d'image. La peinture est toujours plus jaune que la route autour, au soleil comme à l'ombre.
   - Le professeur sautait parfois sur la bande jaunâtre du bord de la route. Correction : la ligne ne peut bouger que de quelques pixels d'une image à l'autre, et un saut plus grand est ignoré.
   - Résultats : sur mini_monaco, **2 tours validés en 73 s**, 95 % du temps à moins de 0,14 du centre. Sur generated_road, 2 minutes sans jamais perdre la ligne.
7. **Enregistrement** sur mini_monaco et generated_road :
   - generated_road est une **route avec une fin** (point 999), pas une boucle. Le professeur la parcourt en entier puis s'arrête au bout.
   - mini_monaco est le circuit le plus difficile pour le professeur. Il sort de temps en temps, vers les points 150, 229 et 257.
   - Pour ne pas apprendre ses erreurs, j'ai ajouté le réglage `trim_before_invalid_s` (1 s) : on jette la dernière seconde avant chaque problème (perte de ligne, sortie, collision). Un test vérifie ce comportement.
   - `monaco_04` a raté dès le départ : le professeur a pris la ligne de départ en damier pour la ligne jaune. Session mise de côté dans `data/excluded_sessions/failed`.
   - Correction : la ligne jaune est maintenant suivie **en chaîne**, ligne d'image par ligne d'image depuis le bas, et la chaîne la plus longue gagne. Si la ligne est perdue trop longtemps, le professeur la recherche sur toute la largeur.
   - `road_04` est sortie de la route dans le désert : seulement 1 233 images valides, mais elles sont gardées car valides. `road_05` est bonne.

8. **Essai du suivi en chaîne** : `monaco_04` s'est cogné dès le départ, et `monaco_05` n'a fait que 1,4 tour (1 943 images valides). Le départ dans la voie de droite se faisait à une position différente à chaque fois, et la bande vert olive du bord de route était prise pour la ligne.
   Décision : lancer tout de suite l'entraînement v4 avec les données déjà là, puis corriger le professeur hors simulateur.
9. **Découpage des données v4** : chaque session est affectée en entier à un seul ensemble (entraînement, validation ou test). La validation et le test contiennent chacun une session de chaque circuit (rapide, lent, warren, monaco, road). Les deux grosses sessions monaco restent à l'entraînement (graine 1588).
10. **Professeur jaune, correction faite hors simulateur** en rejouant les images enregistrées :
    - on exige un vrai jaune (saturation ≥ 0,2), pas un « moins bleu que le mur » ;
    - on choisit la chaîne la plus **intense** : la peinture est brillante (255, 245, 113), le bord de route est olive (135, 133, 63).
    - Résultats : les deux départs ratés trouvent maintenant la bonne ligne. Sur 3 000 images rejouées, il y a 3 fois moins de pertes de ligne.

11. **CNN v4 entraîné sur 4 circuits** ([runs/cnn_multi4_v4](../runs/cnn_multi4_v4)) : 65 111 images d'entraînement, 80 époques en 26 minutes sur la carte graphique.
    - Erreur sur les sessions de test : v2 0,053, v3 0,031, **v4 0,015**. Dans les virages : 0,041, puis 0,039, puis **0,017**.
    - Essais dans le simulateur :

| Circuit | Résultat du CNN v4 |
|---|---|
| Principal ([race_cnn_v2_b](../configs/race_cnn_v2_b.json)) | **3 tours validés sur 3, 41,2 à 41,8 s** (son meilleur temps sûr) |
| warren | **3 tours validés sur 3, environ 48,7 s** |
| mini_monaco | **2 tours validés sur 2, environ 71,5 s**. Il n'est jamais sorti, alors que son professeur sortait parfois. |
| generated_road | 2 min 30 sans aucun problème |
| mountain_track (jamais appris) | ❌ dérive à gauche vers le point 18, monte sur le talus et cale |

12. **v4 sur les 5 autres circuits jamais vus** (generated_track, warehouse, sparkfun_avc, circuit_launch, waveshare) : il échoue sur les 5, comme v2. Il percute un arbre, un pneu, une botte de foin, un avion, une table.
    Conclusion honnête : apprendre 4 circuits rend le CNN très bon **sur ces 4 circuits**. Ça ne suffit pas pour des circuits qui ont un tout autre aspect (intérieur, herbe, hangar). Il faudrait des données de ceux-là aussi.
13. **Vitesse sur le circuit principal avec v4** :
    - Réglage c (2,2 / 1,25) : **sort de la piste** et percute une chaise.
    - Réglage intermédiaire [race_cnn_v4_bc.json](../configs/race_cnn_v4_bc.json) (2,05 / 1,2) : **3 tours validés sur 3, entre 40,1 et 40,5 s**, écart max 0,78. C'est le nouveau meilleur temps sûr (v2 : 42,1 s, v1 : 60,1 s).
14. **CNN + LSTM (réseau récurrent)**, en réponse à ta question sur les RNN :
    - Nouvelle option `model.temporal = "lstm"` dans [racing/model.py](../racing/model.py). Le même CNN regarde chacune des 5 dernières images (une toutes les 0,1 s), puis un LSTM de 128 neurones les lit dans l'ordre du temps.
    - Le format d'entrée ne change pas, donc la préparation des données, le cache sur la carte graphique et le pilotage dans le simulateur marchent sans modification.
    - Un test vérifie que l'ordre des images change bien la réponse.
    - Entraînement « v5 » en cours sur les mêmes 4 circuits, pour comparer honnêtement avec v4.

15. **Résultat CNN + LSTM (v5)** sur les sessions de test, comparé au CNN v4 :
    - Erreur maintenant : **0,0097** contre 0,0148.
    - Ligne droite : 0,0059 contre 0,0114.
    - Virages : 0,0122 contre 0,0174.
    - Le LSTM est environ 35 % plus précis hors simulateur. Il reste à le confirmer en conduite.

## Objectif : faire marcher les 10 circuits

Plan fixé :
1. Définition commune de « ça marche » : 2 minutes ou 1 tour sans collision, caméra seule.
2. Données sur chaque circuit.
3. Un seul entraînement sur tout.
4. Test des 10 circuits avec [tools/final_eval.sh](../tools/final_eval.sh).
5. Rapport final.

16. **Professeur générique « peinture »** (`paint: "white_yellow"`, plus les réglages `white_min`, `white_spread`, `yellow_ratio` et `min_rows`) : il suit la ligne peinte la plus proche de chaque côté, qu'elle soit blanche, jaune ou orange.
    - waveshare : ça marche avec le braquage complet et une vitesse lente, car le circuit est très serré.
    - generated_track : 3,6 tours en 2 minutes au premier essai. Mais ce circuit est **régénéré au hasard à chaque lancement**, et l'ombre des arbres cache la ligne jaune. Il passe donc au professeur CTE (point 17).
17. **Professeur « CTE »** (`guide: "cte"`) : il ne regarde pas l'image. Il suit la distance à la ligne de référence du simulateur, avec un correcteur proportionnel-dérivé.
    - Il sert de professeur avec accès privilégié. Le CNN, lui, apprend et conduit **uniquement avec la caméra**, et il n'y a toujours aucune donnée manuelle. ⚠️ Si le règlement impose un professeur qui « voit » la route, ces circuits devront être présentés à part.
    - Il garde la voie de départ (`cte_center: "spawn"`), parce que la position de départ change d'un lancement à l'autre.
    - Il ignore les sauts impossibles de la référence : la distance ne peut pas changer de plus que le chemin réellement parcouru.
    - warehouse : 7,5 tours en 5 minutes. circuit_launch : 5,25 tours à vitesse lente, mais il touche le mur à vitesse moyenne.
18. **Revalidation par circuit** (`data.revalidate_scenes`) : sur plusieurs circuits, la ligne de référence n'est pas au milieu de la voie. L'enregistreur marquait donc des images parfaites comme non valides. La préparation des données revérifie ces images avec les règles propres au circuit : pas de collision, ligne vue, et une tolérance d'écart adaptée. Un test vérifie que la sécurité est conservée.

## 21 novembre 2026

19. **CNN v6, entraîné sur les 10 circuits** ([runs/cnn_all10_v6](../runs/cnn_all10_v6)) : 138 645 images d'entraînement venant de 44 sessions. Le meilleur modèle est celui de l'époque 10, et l'entraînement s'est arrêté seul à l'époque 22.
    - Test dans le simulateur ([results/final_cnn_all10_v6_run1.jsonl](../results/final_cnn_all10_v6_run1.jsonl)), 2 minutes par circuit, caméra seule :

| Circuit | Sans collision | Tours | Verdict |
|---|---|---|---|
| Principal | 78 s | 1,9 | ✅ (plus d'1 tour), puis touche une fenêtre |
| warren | 120 s | 2,5 | ✅ |
| mini_monaco | 120 s | 1,6 | ✅ |
| generated_road | 120 s | route ouverte | ✅ |
| generated_track | 98 s | 2,6 | ✅ (plus d'1 tour), puis touche un arbre |
| waveshare | 120 s | 6,9 | ✅ |
| warehouse | 120 s | 2,25 | ✅ |
| circuit_launch | 52 s | 0,78 | ❌ mur |
| mountain_track | 120 s | 1,55 | ✅ (une première) |
| sparkfun_avc | 39 s | 0,21 | ❌ botte de foin |

    → **8 circuits sur 10.**
20. **Pourquoi circuit_launch et sparkfun échouaient** : la faute vient des données, pas du CNN.
    - circuit_launch : le professeur CTE touchait le mur dans l'épingle du nœud 108 **à chaque tour**. Comme on jette la seconde qui précède chaque choc, le CNN n'avait **jamais vu** cette épingle. En plus, la voie suivie changeait d'une session à l'autre (selon la position de départ), donc la même image avait des réponses différentes.
    - sparkfun : le professeur se bloquait contre un portique (nœud 352). **30 % du tour (nœuds 353 à 501) n'avait jamais été parcouru.**
21. **Nouveau professeur « poursuite » (`guide: "path"`)**. C'est un professeur avec accès privilégié : il connaît la position de la voiture. Il vise un point 0,8 m plus loin sur une ligne de référence construite à partir de trajets précédents ([tools/build_path.py](../tools/build_path.py)). Il commence donc à tourner **avant** une épingle, et non après.
    - Le CNN, lui, apprend et conduit toujours **uniquement avec la caméra**.
    - Lignes enregistrées : [configs/paths/](../configs/paths/).
    - circuit_launch : **18,5 tours en 15 minutes, 0 collision** (avant : un choc par tour).
    - sparkfun :
      - Exploration des nœuds inconnus avec le professeur CTE (option `path_fallback_cte`).
      - Passage visé au milieu du portique : deux essais avaient touché chacun un poteau, j'ai pris la ligne entre les deux.
      - Puissance augmentée pour la côte du nœud 443.
      - Ligne décalée de 15 cm près du quad sur le tremplin.
      - Dans la spirale, la vraie trajectoire en arc remplace un raccourci impossible.
      - Résultat : **2,2 tours en 5 minutes, 0 collision, 0 arrêt**. C'est le premier tour complet de sparkfun.
    - Les anciennes sessions circuit_launch et sparkfun sont mises de côté dans `data/excluded_sessions/cte_lane_v1`.
22. **Entraînement plus rapide** : option `training.subsample` (on garde 1 image sur 2, car à 20 images par seconde deux images voisines sont presque identiques). Les images tiennent alors entièrement dans la mémoire de la carte graphique 8 Go, ce qui évite le ralentissement de v6 (6 min par époque).

23. **v9 (CNN) et v10 (CNN + LSTM)** : mêmes données que v7, plus 3 sessions sparkfun avec injection de bruit (DART). Erreur de validation : v9 0,047, v10 0,037.
    - Test : 2 essais par circuit, 2 minutes chacun, critère strict (120 s sans aucune collision).

| Circuit | v7 (1 essai) | v9 essai 1 | v9 essai 2 | v10 essai 1 | v10 essai 2 |
|---|---|---|---|---|---|
| Principal | ✅ | ✅ | ❌ cône 28 s | ❌ cône 109 s | ❌ 56 s |
| warren | ✅ | ✅ | ✅ | ✅ | ✅ |
| mini_monaco | ❌ 12 s | ✅ | ✅ | ✅ | ✅ |
| generated_road | ✅ | ✅ | ✅ | ✅ | ✅ |
| generated_track | ✅ | ✅ | ✅ | ❌ 27 s | ❌ 26 s |
| waveshare | ✅ | ✅ | ✅ | ✅ | ✅ |
| warehouse | ❌ 62 s | ❌ 69 s (1,3 tour) | ✅ | ❌ 33 s | ❌ 35 s |
| circuit_launch | ✅ | ❌ 56 s | ✅ | ❌ 56 s | ✅ |
| mountain_track | ✅ | ✅ | ✅ | ✅ | ❌ 116 s |
| sparkfun_avc | ❌ 27 s | ✅ | ✅ | ❌ 33 s | ❌ 33 s |
| **Total** | 7/10 | **8/10** | **9/10** | 5/10 | 6/10 |

    - **Le bruit DART a réparé sparkfun** pour le CNN (2 sur 2). Le LSTM, malgré une meilleure erreur de validation, conduit moins bien : il sort au même endroit à chaque essai. **On garde le CNN.**
    - Échecs restants de v9 : la voiture s'écarte de la ligne et ne corrige pas (circuit_launch : écart −4,3 après l'épingle, alors que le professeur n'est jamais allé au-delà de −2). C'est le même problème « pas de données de retour » que sparkfun avant DART.
24. **Bruit DART sur circuit_launch, warehouse et le circuit principal** : 7 sessions propres gardées (claunch_d2, d4, d5 ; whouse_d1, d2 ; main_d2, d3).
    - 3 sessions mises de côté (`data/excluded_sessions/failed`), parce que la poussée sur le volant arrivait dans un virage serré.
    - Nouvelle règle `perturb.max_teacher_steering` (0,6) : on ne pousse jamais quand le professeur tourne déjà fort.
    - Modèle v11 = v9 + ces 7 sessions ([tools/train_eval_v11.sh](../tools/train_eval_v11.sh)).

25. **v11 : 10 circuits sur 10** au premier essai, et 9 sur 10 au deuxième. Au total, 19 essais réussis sur 20.
    - Résultats : [results/final_cnn_all10_v11_r1.jsonl](../results/final_cnn_all10_v11_r1.jsonl), [r2](../results/final_cnn_all10_v11_r2.jsonl).
    - Le seul échec est generated_track au 2ᵉ essai : arbre touché à 29 s. Ce circuit est **tiré au hasard à chaque lancement**, et celui-ci avait un double virage en S. Le CNN a zigzagué dès le départ, puis est sorti large.
    - Erreur de validation 0,043, contre 0,047 pour v9.
    - Pour fiabiliser ce circuit : 3 sessions generated_track avec bruit DART (gtrack_d1 à d3), chacune sur un tracé différent, puis le modèle v12.

26. **Sortie de route invisible au compteur de collisions (waveshare)** : l'utilisateur a vu la voiture rouler sur le logo, hors de la route. Le simulateur ne signale rien, parce qu'il n'y a que de la peinture.
    - Cause : le professeur « peinture » de waveshare zigzaguait lui-même (écart ±4 à 5). Le réseau copiait ses erreurs.
    - Correction : nouvelle ligne de référence ([configs/paths/waveshare.json](../configs/paths/waveshare.json)), professeur « poursuite », et 3 nouvelles sessions (wshare_p1, d1, d2). Chacune fait environ 21 tours en 5 minutes, avec un écart d'environ ±1 et 0 collision.
    - Les anciennes sessions sont mises de côté dans `data/excluded_sessions/wshare_paint_v1`.
    - `tools/build_path.py --upsample K` : ligne plus fine sur les circuits qui ont peu de nœuds.
27. **Nouvelle mesure honnête** ([tools/offtrack_check.py](../tools/offtrack_check.py)) : on compte un essai comme réussi seulement s'il n'y a **aucune collision ET** pas plus d'1 s hors de la route.
    - Hors de la route = plus de 30 cm de tout endroit où un professeur est passé.
    - Pour les circuits aléatoires (generated_road et generated_track), on compare plutôt l'écart du simulateur à celui des professeurs.
    - Avec cette mesure, v11 n'a que 8 et 7 circuits propres sur 10 (au lieu de 10 et 9), et v12 en a 8 sur 10 au 1ᵉʳ essai.
28. **100 % deep learning** (demande de l'enseignant). Option `model.throttle_output` : le réseau a une 4ᵉ sortie, la « pédale » (accélérateur moins frein), apprise sur ce que faisait le professeur.
    - Avec `control.throttle_source = "network"`, le volant, l'accélérateur et le frein viennent **tous** du réseau. Il ne reste qu'un plafond de sécurité.
    - Un test vérifie que l'effet miroir inverse le volant mais pas la pédale. 59 tests passent.
29. **Carte graphique** : lots de 256 images (au lieu de 64), avec un pas d'apprentissage de 0,0006.
    - La carte est à 90-94 % pendant l'entraînement (mesure toutes les 15 s, `logs/gpu_v13.log`).
    - L'entraînement prend 15 min au lieu de 45.
    - v13 : erreur de volant en validation 0,038.
    - Le pipeline ([tools/pipeline_v13.sh](../tools/pipeline_v13.sh)) tourne dans un processus détaché, pour survivre à un redémarrage de l'éditeur.

30. **v13 testé** (critère strict).
    - Avec la règle de vitesse : **9/10**. waveshare est réparé : la voiture reste à 1 cm de la ligne du professeur.
    - En mode 100 % réseau : 7/10. La voiture cale dans la côte de mountain, touche un objet à 119,9 s sur sparkfun, et sort de la route sur generated_track.
    - Cause pour mountain : le professeur n'a jamais été lent dans la côte. Le réseau ne savait donc pas qu'il faut accélérer plus quand on est lent.
31. **DART sur la pédale** : options `perturb.pedal`, `pedal_share` et `pedal_skip_nodes` dans [racing/drive.py](../racing/drive.py).
    - Pendant 1 s, on coupe l'accélérateur ou on le renforce, et on garde comme label ce que le professeur voulait.
    - 12 sessions prévues (configs `configs/tracks/expert_pedal_<circuit>.json`), **8 gardées**.
    - Écartées par [tools/check_session.py](../tools/check_session.py) :
      - sparkfun ×2 : la voiture calait dans la côte quand on coupait ;
      - generated_road : arrêt normal au bout de la route ;
      - mount_q1 : deux lancements du simulateur se sont chevauchés.
32. **v14** = v13 + ces 8 sessions (64 sessions, 250 035 exemples d'entraînement). Erreur de volant en validation : **0,0345**.
    - **100 % réseau** : **10/10** au 1ᵉʳ passage, **8/10** au 2ᵉ (generated_track touché à 86 s après 2 tours ; sparkfun 6 s à 31-47 cm de la trajectoire).
    - **Règle de vitesse** : **10/10**.
    - generated_road, 2ᵉ passage : le CTE du simulateur était cassé (233 à 300), mais les photos montrent la voiture dans sa voie. [tools/offtrack_check.py](../tools/offtrack_check.py) détecte maintenant ce cas, et refuse aussi les essais interrompus.
    - Un 3ᵉ passage n'est pas retenu : la machine était ralentie (réseau 3,5× plus lent, 2 essais coupés par les sécurités). Le détail est dans [docs/RAPPORT_COMPLET.md](RAPPORT_COMPLET.md), partie 14.
33. **Modèle final : v14** ([runs/cnn_all10_v14/best.pt](../runs/cnn_all10_v14/best.pt)), avec les configs `configs/tracks/cnn_v14n_<circuit>.json` (100 % réseau).
    - Le rapport complet, pour comprendre tout le projet, est dans [docs/RAPPORT_COMPLET.md](RAPPORT_COMPLET.md).

