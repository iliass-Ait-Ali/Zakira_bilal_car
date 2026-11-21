# Rapport complet : une voiture autonome par deep learning

**Projet : Zakira_bilal_car.** Simulateur DonkeyCar (SDSandbox), réseau de neurones convolutif (CNN) entraîné avec PyTorch sur une carte graphique NVIDIA RTX 4060.

Ce rapport explique **tout** le projet, depuis les idées de base jusqu'aux détails du code.

- **Pour comprendre le projet**, lis les parties 1 à 5.
- **Pour savoir comment il a été construit et pourquoi**, lis les parties 6 à 12.
- **Pour le refaire ou le modifier**, lis les parties 13 à 16.

---

## Sommaire

0. [Résumé en une page](#0-résumé-en-une-page)
1. [Le but et les règles](#1-le-but-et-les-règles)
2. [Les notions de base, expliquées simplement](#2-les-notions-de-base-expliquées-simplement)
3. [Les outils : simulateur, Python, carte graphique](#3-les-outils--simulateur-python-carte-graphique)
4. [Le chemin d'une image : du simulateur au volant](#4-le-chemin-dune-image--du-simulateur-au-volant)
5. [Le réseau de neurones, couche par couche](#5-le-réseau-de-neurones-couche-par-couche)
6. [Les données : comment on fabrique les exemples](#6-les-données--comment-on-fabrique-les-exemples)
7. [Les professeurs automatiques](#7-les-professeurs-automatiques)
8. [L'entraînement](#8-lentraînement)
9. [Le contrôle : du réseau aux pédales](#9-le-contrôle--du-réseau-aux-pédales)
10. [Les 10 circuits, un par un](#10-les-10-circuits-un-par-un)
11. [Les techniques qui ont tout changé](#11-les-techniques-qui-ont-tout-changé)
12. [L'histoire du projet, version par version](#12-lhistoire-du-projet-version-par-version)
13. [Comment on mesure la réussite](#13-comment-on-mesure-la-réussite)
14. [Résultats finaux](#14-résultats-finaux)
15. [Ce qu'on a appris (les leçons)](#15-ce-quon-a-appris-les-leçons)
16. [Limites et idées pour aller plus loin](#16-limites-et-idées-pour-aller-plus-loin)
17. [Guide pratique : installer, conduire, entraîner](#17-guide-pratique--installer-conduire-entraîner)
18. [Carte des fichiers du projet](#18-carte-des-fichiers-du-projet)
19. [Glossaire](#19-glossaire)
20. [Questions fréquentes](#20-questions-fréquentes)

---

## 0. Résumé en une page

**Le problème.** Une petite voiture du simulateur DonkeyCar doit rouler toute seule sur les 10 circuits du simulateur. Elle ne doit utiliser **que sa caméra** (et sa vitesse), et elle doit être conduite **par un réseau de neurones (deep learning)**. C'est la règle donnée par l'enseignant.

**La méthode : l'apprentissage par imitation** (en anglais *behavioral cloning*). C'est la méthode classique de DonkeyCar et de la voiture autonome de NVIDIA (2016).
1. Quelqu'un conduit la voiture, et on enregistre à chaque instant **l'image de la caméra** et **ce que le conducteur a fait** (volant, accélérateur, frein).
2. On montre des centaines de milliers de ces exemples à un réseau de neurones. Il apprend à deviner, **à partir de l'image seule**, ce que le conducteur aurait fait.
3. Pendant le test, le conducteur est retiré. **Le réseau conduit seul**, avec la caméra.

**Qui conduit pour fabriquer les exemples ?** Pas un humain, mais des **professeurs automatiques** : des petits programmes qui savent conduire chacun sur leur circuit. Certains suivent les lignes peintes vues par la caméra. D'autres connaissent la position exacte de la voiture donnée par le simulateur. Ces professeurs servent **seulement** à produire les exemples. **Pendant le test, seul le réseau conduit.**

**Le réseau.** Un CNN (réseau convolutif) de 232 802 paramètres.
- Entrée : une image de 160 × 96 pixels et la vitesse.
- Sorties : le braquage **maintenant**, **dans 0,25 s** et **dans 0,5 s**, plus la **pédale** (accélérateur moins frein).

**Les techniques décisives.**
1. **Des données de chaque circuit.** Un réseau entraîné sur 4 circuits échoue sur tous les autres.
2. **De bons professeurs.** Le réseau copie aussi les défauts du professeur. Sur waveshare, le professeur sortait de la route, et le réseau aussi.
3. **L'injection de bruit (DART).** Pendant l'enregistrement, on pousse de temps en temps le volant (ou la pédale) de travers, mais on enregistre ce que le professeur *voulait* faire. Le réseau apprend ainsi à **revenir** sur la trajectoire quand il s'en écarte. Sans cela, la moindre erreur le fait sortir.
4. **Une mesure honnête.** On compte comme réussi un essai où il n'y a **aucune collision et aucune sortie de route**, pendant 2 minutes.

**Résultat** (détails dans la [partie 14](#14-résultats-finaux)). Le modèle final s'appelle **v14**.
- **Conduit à 100 % par le réseau** (volant, accélérateur, frein), il réussit **10 circuits sur 10** au 1ᵉʳ essai et **8 sur 10** au 2ᵉ, soit 18 essais sur 20.
- **Avec la petite règle de vitesse**, il réussit **10 sur 10**.

Le critère est strict : aucune collision **et** pas de sortie de route pendant 2 minutes. Les deux échecs sont un contact après 86 s sur generated_track, le circuit au tracé aléatoire, et 6 s un peu à côté de la trajectoire sur sparkfun, sans collision.

---

## 1. Le but et les règles

### 1.1 Le but

Faire rouler une voiture **toute seule** sur les **10 circuits** du simulateur DonkeyCar :

| # | Nom dans le simulateur | Description courte |
|---|---|---|
| 1 | `roboracingleague_1` | le circuit « principal », intérieur, lignes blanches |
| 2 | `warren` | grande salle, lignes blanches, chaises et tables autour |
| 3 | `mini_monaco` | circuit urbain, ligne jaune au centre, murets |
| 4 | `generated_road` | longue route dans le désert, ligne jaune pointillée (route avec une fin) |
| 5 | `generated_track` | circuit dans l'herbe avec des arbres, **tracé tiré au hasard** |
| 6 | `waveshare` | tout petit circuit sur un tapis imprimé (logo « WAVESHARE ») |
| 7 | `warehouse` | entrepôt, pneus et barrières |
| 8 | `circuit_launch` | hangar avec murs en plastique, une épingle très serrée |
| 9 | `mountain_track` | route de montagne avec des côtes |
| 10 | `sparkfun_avc` | parcours d'obstacles : portique, spirale, côte, tremplin, bottes de foin |

### 1.2 Les règles

1. **Deep learning seulement** (règle de l'enseignant). C'est un **réseau de neurones** qui conduit la voiture pendant le test.
2. **Caméra seulement** pendant la conduite. Le réseau reçoit l'image et la vitesse. Il **ne reçoit pas** la position de la voiture sur la carte.
3. **Critère de réussite** : sur un circuit, un essai de 2 minutes est réussi s'il n'y a **aucune collision** et si la voiture **ne sort pas de la route** plus d'une seconde (voir [partie 13](#13-comment-on-mesure-la-réussite)).

### 1.3 Ce qui est « deep learning » et ce qui ne l'est pas

| Partie | Qui le fait ? | Deep learning ? |
|---|---|---|
| Tourner le volant pendant le test | le réseau CNN | ✅ oui |
| Accélérer et freiner pendant le test | le réseau CNN (sortie « pédale ») dans le mode `throttle_source: "network"` ; ou une petite règle de vitesse dans l'ancien mode | ✅ oui dans le mode réseau |
| Fabriquer les exemples d'entraînement | les professeurs automatiques | ❌ non, ce sont les « moniteurs d'auto-école » |
| Préparer les images (découpe, réduction) | du code classique | ❌ (prétraitement normal) |

On peut comparer avec une auto-école. Le professeur conduit et l'élève regarde. Ensuite l'élève conduit seul, et c'est lui qu'on note. **Le réseau est l'élève.** Un humain au clavier pourrait aussi servir de professeur : le code le permet (`--mode manual`). Les professeurs automatiques conduisent simplement plus régulièrement et ne se fatiguent pas.

---

## 2. Les notions de base, expliquées simplement

### 2.1 Un réseau de neurones

Un réseau de neurones est une grosse **fonction mathématique avec des réglages** (les « poids » ou « paramètres »). On lui donne une entrée, ici une image, et il calcule une sortie, ici un braquage.

Au début, les réglages sont aléatoires, donc les réponses sont mauvaises. **L'entraînement** ajuste petit à petit les réglages pour que les réponses ressemblent aux bonnes réponses des exemples.

### 2.2 L'apprentissage supervisé

« Supervisé » veut dire qu'on a, pour chaque exemple, **la bonne réponse** (le « label »). Ici, le label est ce qu'a fait le professeur : par exemple « braquer à 0,3 vers la droite ».

### 2.3 La convolution (CNN)

Une image, c'est un tableau de pixels. Une **convolution** fait glisser un petit filtre (par exemple 5 × 5 pixels) sur toute l'image, pour y repérer un motif : un bord, une ligne, une couleur.
- Les premières couches détectent des motifs simples, comme les bords des lignes blanches.
- Les couches suivantes combinent ces motifs en choses plus utiles, comme « la ligne tourne à gauche au loin ».

Un **CNN** (*Convolutional Neural Network*) est une suite de convolutions, suivie de quelques couches « denses » qui prennent la décision.

### 2.4 La perte (loss)

La **perte** est un nombre qui dit « à quel point le réseau se trompe ». Si le professeur a braqué à 0,30 et que le réseau dit 0,25, l'erreur est de 0,05. On utilise une perte de **Huber** :
- elle se comporte comme le carré de l'erreur pour les petites erreurs ;
- elle se comporte comme la valeur absolue pour les grandes erreurs, ce qui la rend moins sensible aux exemples bizarres.

### 2.5 La descente de gradient

Pour réduire la perte, on calcule dans quel sens bouger chaque réglage : c'est le **gradient**. Puis on le bouge un tout petit peu dans ce sens. La taille de ce petit pas s'appelle le **pas d'apprentissage** (*learning rate*). On répète des centaines de milliers de fois. L'algorithme utilisé s'appelle **AdamW**.

### 2.6 Époque, lot (batch)

- Un **lot** (*batch*) est un paquet d'exemples traités ensemble, ici 256.
- Une **époque** est un passage complet sur tous les exemples d'entraînement. On fait 60 époques.

### 2.7 Entraînement, validation, test

On sépare les données en trois groupes, **par sessions entières** :
- **Entraînement** : les exemples que le réseau apprend.
- **Validation** : des exemples jamais appris. On s'en sert pour garder la meilleure version du réseau (`best.pt`) et pour arrêter quand il ne progresse plus.
- **Test** : des exemples jamais utilisés pendant l'entraînement, gardés pour une vérification finale.

**Pourquoi par sessions entières ?** Deux images qui se suivent (1/20ᵉ de seconde d'écart) sont presque identiques. Si l'une était en entraînement et l'autre en validation, le réseau « tricherait » : la mesure serait trop optimiste.

### 2.8 Le sur-apprentissage (overfitting)

C'est quand le réseau apprend **par cœur** les exemples au lieu de comprendre la règle générale. Il est alors excellent sur les exemples et mauvais ailleurs. On le combat de plusieurs façons :
- beaucoup de données variées ;
- l'**augmentation** : on modifie un peu les images au hasard ;
- le **dropout** : on éteint au hasard 15 % des neurones pendant l'apprentissage ;
- l'arrêt anticipé.

### 2.9 L'imitation et son grand piège

Le réseau apprend seulement à partir de ce que le professeur a montré. Or un bon professeur reste toujours au milieu de la route : le réseau ne voit donc **jamais** à quoi ressemble « être un peu à côté », ni ce qu'il faut faire dans ce cas.

Pendant le test, le réseau fait forcément de petites erreurs. Il se retrouve un peu à côté, dans une situation qu'il n'a jamais vue, et se trompe encore plus. Les erreurs **s'accumulent** et la voiture sort.

C'est le problème le plus connu de l'apprentissage par imitation, appelé en anglais *covariate shift* ou *compounding errors*. La solution utilisée ici s'appelle **DART** (partie 11.1).

### 2.10 La carte graphique (GPU)

Un réseau de neurones fait énormément de multiplications. Une carte graphique en fait des milliers en même temps, alors qu'un processeur normal (CPU) en fait quelques-unes. Ici, l'entraînement prend **15 minutes** sur la RTX 4060, contre plusieurs heures sur le processeur.

---

## 3. Les outils : simulateur, Python, carte graphique

### 3.1 Le simulateur DonkeySim (SDSandbox)

- C'est un programme Windows fait avec le moteur de jeu Unity : `DonkeySimWin/DonkeySimWin/donkey_sim.exe`.
- Il contient les 10 circuits.
- Il communique avec notre programme Python par le **réseau local** (adresse `127.0.0.1`, port `9091`), avec des messages au format JSON.
- Environ 20 fois par seconde, il envoie la **télémétrie** :
  - l'image de la caméra, encodée en PNG, 255 × 255 pixels en couleur, avec un champ de vision de 150° ;
  - la vitesse ;
  - la position `pos_x` / `pos_z` et l'angle `yaw` ;
  - le **CTE** (*cross-track error* : écart par rapport au chemin de référence du simulateur) ;
  - le numéro du point de passage `activeNode` ;
  - les collisions (`hit`).
- Notre programme lui renvoie à chaque image : **volant** (−1 à +1), **accélérateur** (0 à 1), **frein** (0 à 1).

**Important** : la position, le CTE et `activeNode` ne sont **jamais** donnés au réseau. Ils servent seulement :
1. à certains professeurs, pour fabriquer les exemples ;
2. à **mesurer** les résultats : collisions, sortie de route, tours.

### 3.2 Python et les bibliothèques

- **Python 3.13**, dans un environnement virtuel `.venv`.
- **PyTorch 2.10** (version CUDA 12.8) pour le réseau de neurones.
- **NumPy** pour les tableaux de nombres, **Pillow** et **OpenCV** pour les images, **Matplotlib** pour les graphiques.
- Le client réseau du simulateur (`SDClient`, licence MIT) est recopié et corrigé dans `racing/vendor/` et `racing/transport.py`. La version installée par pip était incomplète.

### 3.3 La machine

- Ordinateur portable Lenovo Legion, Windows 11.
- Carte graphique **NVIDIA GeForce RTX 4060 Laptop**, 8 Go de mémoire.
- Le simulateur et l'entraînement ne peuvent pas tourner en même temps sans gêne. Le test de conduite a besoin de décisions rapides : moins de 0,3 s par image, sinon le programme s'arrête par sécurité.

---

## 4. Le chemin d'une image : du simulateur au volant

Voici ce qui se passe **20 fois par seconde** pendant la conduite par le réseau (fichier `racing/drive.py`, mode `--mode cnn`).

```
 SIMULATEUR                                    PROGRAMME PYTHON
 ──────────                                    ────────────────
 caméra 255×255 ──(JSON + PNG, port 9091)──►  1. réception + décodage (racing/transport.py)
 vitesse, télémétrie                                        │
                                               2. prétraitement (racing/preprocess.py)
                                                  découpe : on enlève 30 % en haut (ciel, murs)
                                                            et 5 % en bas (capot)
                                                  réduction à 160 × 96 pixels
                                                  valeurs ramenées entre −1 et +1
                                                            │
                                               3. réseau CNN (racing/model.py, racing/inference.py)
                                                  entrée : image + vitesse/10
                                                  sortie : [braquage maintenant,
                                                            braquage dans 0,25 s,
                                                            braquage dans 0,5 s,
                                                            pédale]
                                                            │
                                               4. contrôle (racing/control.py)
                                                  volant : braquage lissé (filtre 0,06 s,
                                                           vitesse de rotation limitée)
                                                  pédale > 0 → accélérateur ; pédale < −0,05 → frein
                                                            │
 volant, accélérateur, frein ◄─────────────── 5. envoi de la commande
```

Il y a aussi des **sécurités**, qui ne prennent jamais de décision de conduite :
- Si une image arrive trop vieille (plus de 0,3 s), le programme freine et s'arrête.
- Si le réseau donne un nombre invalide, il freine.
- Si la connexion se coupe, il s'arrête.

---

## 5. Le réseau de neurones, couche par couche

Fichier : `racing/model.py`, classe `RacingCNN`.

### 5.1 L'architecture

L'architecture s'inspire du réseau PilotNet de NVIDIA (2016), en plus petit.

| Étape | Opération | Taille du résultat |
|---|---|---|
| Entrée | image couleur, 3 canaux (rouge, vert, bleu) | 3 × 96 × 160 |
| Conv 1 | 24 filtres 5×5, pas de 2, activation ELU | 24 × 48 × 80 |
| Conv 2 | 36 filtres 5×5, pas de 2, ELU | 36 × 24 × 40 |
| Conv 3 | 48 filtres 5×5, pas de 2, ELU | 48 × 12 × 20 |
| Conv 4 | 64 filtres 3×3, pas de 1, ELU | 64 × 12 × 20 |
| Conv 5 | 64 filtres 3×3, pas de 1, ELU | 64 × 12 × 20 |
| Moyenne par zone | moyenne sur des carrés 4×4 : grille de **3 × 5 zones** | 64 × 3 × 5 = 960 nombres |
| + vitesse | on ajoute la vitesse divisée par 10 | 961 nombres |
| Dense 1 | 100 neurones, ELU, dropout 15 % | 100 |
| Dense 2 | 50 neurones, ELU, dropout 15 % | 50 |
| Sortie | 4 neurones, activation `tanh` (résultat entre −1 et +1) | 4 |

**Total : 232 802 paramètres** (232 751 sans la sortie pédale).

### 5.2 Les choix de conception, et pourquoi

- **La grille 3 × 5 au lieu d'une moyenne globale.** Une moyenne sur toute l'image ferait perdre **où** se trouve la ligne : à gauche ou à droite. Or c'est justement l'information la plus importante pour tourner le volant. La grille garde une position grossière.
- **La vitesse en entrée.** À 1 m/s et à 2 m/s, il ne faut pas tourner pareil ni au même moment.
- **Trois braquages : maintenant, dans 0,25 s, dans 0,5 s.** Le réseau apprend à **anticiper**. Le contrôleur de l'ancien mode s'en sert pour ralentir **avant** un virage.
- **ELU** (*Exponential Linear Unit*) : une activation douce, qui laisse passer un peu de signal négatif. Elle apprend souvent mieux que ReLU sur ce genre de petit réseau.
- **`tanh` en sortie** : les commandes du simulateur sont bornées entre −1 et 1, et tanh garantit des sorties dans cet intervalle.
- **Petit réseau** : 0,23 million de paramètres. Il est vite entraîné et il décide en quelques millisecondes. C'est indispensable à 20 images par seconde.

### 5.3 La sortie « pédale » (option `model.throttle_output`)

Ajoutée pour que **tout** soit appris par le réseau. C'est une 4ᵉ sortie : la **pédale** = accélérateur − frein du professeur, entre −1 et +1.
- Si elle est positive, on accélère d'autant.
- Si elle est sous −0,05, on freine.
- **Détail important** : quand on retourne une image en miroir pendant l'apprentissage, le braquage change de signe mais **pas la pédale**. Un virage à gauche devient un virage à droite, mais il faut toujours accélérer pareil. Un test automatique vérifie ce point.

### 5.4 La variante CNN + LSTM (testée, pas retenue)

Option `model.temporal = "lstm"`. Le même CNN regarde les **5 dernières images** (une toutes les 0,1 s), puis un **LSTM** de 128 neurones les lit dans l'ordre. Un LSTM est un réseau récurrent qui a une mémoire, et le modèle voit donc le mouvement. Il a 707 631 paramètres.
- Sur 4 circuits (v5), il était 35 % plus précis hors simulateur et un peu plus rapide sur le circuit.
- Sur les 10 circuits (v10), il avait **la meilleure erreur de validation (0,037 contre 0,047)** mais il **conduisait moins bien** : 10 essais réussis sur 20, contre 17 pour le CNN.
- **Leçon** : une bonne note sur les exemples enregistrés ne garantit pas une bonne conduite. Il faut toujours tester dans le simulateur.

---

## 6. Les données : comment on fabrique les exemples

### 6.1 L'enregistrement (une « session »)

Commande type :

```
python -m racing.drive --mode baseline --connect --record --baseline-validated --group claunch_d4 --seconds 300
```

Pendant 5 minutes, le professeur conduit et on enregistre (fichiers `racing/recording.py` et `racing/drive.py`) :

```
data/sessions/<identifiant>/
   session.json     → réglages complets, caméra, nom du groupe (ex. "claunch_d4")
   records.jsonl    → une ligne par image : heure, commande du professeur, vitesse,
                      télémétrie, "label_valid", "recovery", commande réellement exécutée
   images/00000000.png, 00000001.png, ...   → les images de la caméra
```

Une session de 5 minutes fait environ **5 800 images**. Le projet en compte une soixantaine, soit plus de 300 000 images.

### 6.2 Quand un exemple est-il « valide » ?

Un exemple n'est utilisé pour apprendre que s'il est **propre**. Il ne l'est pas dans les cas suivants :
- il y a une collision ;
- le professeur a perdu la ligne (`lane_lost`) ;
- la voiture est trop loin du chemin de référence ;
- l'image est trop vieille ou il y a un trou dans le temps (plus de 0,15 s).

**On jette aussi la seconde qui précède chaque problème** (`trim_before_invalid_s = 1.0`). Ce sont souvent les mauvaises décisions qui ont **causé** le problème : on ne veut pas les apprendre.

**Revalidation par circuit** (`revalidate_scenes`). Sur certains circuits, la ligne de référence du simulateur n'est pas au milieu de la route, ou elle fait des sauts. La préparation revérifie alors chaque image avec une tolérance adaptée au circuit (`cte_reference`).

### 6.3 La préparation (`racing/data.py`)

```
python -m racing.data --sessions data/sessions --out data/prepared/all10_v7 --config configs/train_v14_all.json
```

1. Lit toutes les sessions et vérifie la caméra, les heures et les valeurs.
2. Découpe chaque session en **séquences continues** (coupées à chaque problème).
3. Pour chaque image, construit le **label** :
   - le braquage du professeur maintenant ;
   - le braquage 0,25 s plus tard ;
   - le braquage 0,5 s plus tard ;
   - la pédale.

   Si l'instant futur n'existe pas (fin de séquence), ce label est **masqué**, c'est-à-dire ignoré, et jamais inventé.
4. Répartit les **sessions entières** en entraînement, validation et test. La validation contient une session de chaque circuit. Les groupes sont choisis dans la config (`val_groups`, `test_groups`).
5. Vérifie qu'aucune image n'apparaît dans deux ensembles (contrôle par empreinte des pixels).
6. Écrit `train.jsonl`, `val.jsonl`, `test.jsonl`, `dataset.json` (le résumé) et `rejected.json` (ce qui a été refusé, et pourquoi).

Pour v13 : **203 508** exemples d'entraînement, **59 227** de validation et **23 290** de test, venant de 56 sessions.

### 6.4 L'équilibrage

Sur un circuit, on va **tout droit** la plupart du temps. Si on ne fait rien, le réseau apprend surtout « aller tout droit ». Chaque exemple reçoit donc un poids selon la force du virage :
- les virages sont tirés plus souvent (correction en racine carrée, plafonnée à 4×) ;
- les exemples de « retour sur la ligne » (`recovery`) comptent 1,5×.

### 6.5 L'augmentation des données

À chaque passage, chaque image est un peu modifiée au hasard, pour que le réseau ne dépende pas de détails inutiles :
- **luminosité** ±15 % et **contraste** ±15 % (lumière différente) ;
- **miroir** gauche-droite (une fois sur deux), avec le braquage inversé ;
- un peu de **bruit** sur les pixels.

### 6.6 Le cache sur la carte graphique

Lire et décoder 200 000 fichiers PNG à chaque époque est très lent. Le programme prépare donc toutes les images **une seule fois** (découpe et réduction), les range dans un gros fichier `data/cache/images_<code>.npy`, et le charge **entièrement dans la mémoire de la carte graphique** (option `--gpu-data`). Les modifications aléatoires sont faites directement sur la carte (fichier `racing/gpu_data.py`).
- On garde **1 image sur 2** (`subsample = 2`). À 20 images par seconde, deux images voisines sont presque identiques, et c'est ce qui permet au cache de tenir dans les 8 Go de la carte.

---

## 7. Les professeurs automatiques

Fichier : `racing/baseline.py`, classe `BaselinePilot`. Chaque circuit a le professeur qui lui convient le mieux. Ils sont choisis dans la config, avec l'option `baseline.guide`.

| Professeur | Comment il conduit | Voit quoi ? | Circuits |
|---|---|---|---|
| **Lignes blanches** (par défaut) | cherche les deux lignes blanches dans le bas de l'image et vise le milieu. Avec `single_line`, si une ligne disparaît, il suit l'autre en se souvenant de la largeur de la voie | caméra | principal, warren |
| **Ligne jaune** (`guide: "yellow"`) | suit la ligne jaune centrale, **en chaîne** de bas en haut de l'image ; refuse les sauts ; accepte les pointillés (`lost_grace_s`) | caméra | mini_monaco, generated_road |
| **CTE** (`guide: "cte"`) | régulateur PD sur l'écart au chemin de référence du simulateur | position (privilégié) | generated_track, warehouse, mountain_track |
| **Poursuite** (`guide: "path"`) | vise un point 0,8 m plus loin (0,5 m sur waveshare) sur une **ligne enregistrée** (`configs/paths/<circuit>.json`) : c'est la méthode classique du *pure pursuit* | position (privilégié) | circuit_launch, sparkfun_avc, waveshare |

Tous les professeurs règlent leur vitesse de la même façon. Ils visent une vitesse plus basse dans les virages, puis appliquent un régulateur PI : accélérateur = petite base + correction proportionnelle + correction intégrale.

**« Privilégié »** veut dire que le professeur utilise une information que le réseau n'a pas : la position donnée par le simulateur. C'est autorisé, parce que **le professeur ne conduit jamais pendant le test**. Ce n'est qu'un moyen de produire de bons exemples. Dans une vraie voiture, on ferait la même chose avec un GPS précis, ou avec un humain.

### 7.1 Comment on fabrique une ligne pour le professeur « poursuite »

Outil : `tools/build_path.py`.
1. On fait rouler un professeur une première fois, puis on enregistre les positions.
2. Pour chaque point de passage du simulateur, on prend la **position médiane** de la voiture.
3. On bouche les trous par interpolation, puis on lisse la ligne (option `--upsample` pour une ligne plus fine).
4. On corrige à la main les endroits délicats. Sur sparkfun : passer au milieu du portique, éviter le quad, suivre l'arc réel de la spirale.

Pour **waveshare**, on a gardé seulement les moments où la voiture était **exactement sur le chemin du simulateur** (CTE proche de 0). On a ensuite fait la moyenne par angle autour du centre du circuit.

---

## 8. L'entraînement

Fichier : `racing/train.py`. Commande type :

```
python -m racing.train --config configs/train_v14_all.json --dataset data/prepared/all10_v7 --out runs/cnn_all10_v14 --cache-dir data/cache --gpu-data --start-training
```

Le drapeau `--start-training` est obligatoire : il évite de lancer un entraînement par erreur.

### 8.1 Les réglages (modèle v13/v14)

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Optimiseur | AdamW | standard, stable |
| Pas d'apprentissage | 0,0006, puis baisse en cosinus jusqu'à 0,00001 | grands pas au début, petits pas à la fin |
| Lot (batch) | 256 images | remplit la carte graphique (90-94 %) ; avec 64 images, elle était sous-utilisée |
| Époques | 60 maximum, arrêt si 15 époques sans progrès | |
| Perte | Huber (beta = 0,1), masquée | robuste ; ignore les labels futurs absents |
| Poids des 4 sorties | 0,6 / 0,25 / 0,15 / 0,3 | le braquage actuel compte le plus |
| Régularisation | weight decay 0,0001, dropout 15 %, gradient limité à 1 | contre le sur-apprentissage et les explosions |
| Précision mixte (AMP) | oui | calculs en demi-précision : 2× plus rapide sur la carte |
| Graine aléatoire | 42, mode déterministe | même résultat si on recommence |

### 8.2 Ce que produit un entraînement

Dossier `runs/<nom>/` :
- `best.pt` : **le meilleur modèle**, celui qui a la plus petite perte de validation. C'est lui qu'on utilise pour conduire.
- `last.pt` : le modèle à la dernière époque.
- `metrics.csv` : pour chaque époque, la perte d'entraînement et de validation, l'erreur moyenne et l'erreur dans les virages.
- `config.json`, `dataset.json`, `environment.json`, `checkpoint_hashes.json` : de quoi savoir **exactement** comment le modèle a été fait.

Chaque modèle contient aussi son **contrat** : la caméra, le prétraitement et l'architecture. Au chargement, le programme vérifie que la config de conduite a exactement le même contrat. Un modèle entraîné sur des images 160 × 96 ne peut pas être utilisé par erreur avec d'autres images.

### 8.3 Durée

- 60 époques en **15 minutes** (lots de 256, données dans la carte graphique).
- La préparation des données prend environ 10 minutes, sur le processeur.

---

## 9. Le contrôle : du réseau aux pédales

Fichier : `racing/control.py`, classe `Controller`.

### 9.1 Le volant (les deux modes)

On prend le braquage « maintenant » prédit par le réseau. On le lisse légèrement (filtre de 0,06 s), et sa vitesse de changement est limitée à 5 unités par seconde. Ça évite les à-coups.

### 9.2 L'accélérateur et le frein

**Mode réseau** (`control.throttle_source = "network"`) : **100 % deep learning**.
- Pédale > 0 → accélérateur = pédale (avec un plafond de sécurité réglable).
- Pédale < −0,05 → frein = −pédale.

**Mode règle** (l'ancien mode, `throttle_source = "rule"`) :
1. On calcule une « demande de virage » = la plus grande valeur absolue des 3 braquages prédits.
2. Vitesse visée = vitesse en ligne droite / (1 + 2 × demande), mais jamais moins que la vitesse en virage. Elle ne peut monter que de 1 m/s par seconde.
3. Un régulateur PI (proportionnel + intégral) choisit l'accélérateur pour atteindre cette vitesse. Il freine si on va trop vite (avec une hystérésis pour ne pas osciller).

Ce mode garde le **volant** appris par le réseau, mais la **vitesse** suit une règle écrite à la main. C'est pour cela qu'on a ajouté le mode réseau.

### 9.3 Les vitesses par circuit (fichiers `configs/tracks/cnn_v14*_<circuit>.json`)

Chaque circuit a son fichier de conduite. On y règle les vitesses (pour le mode règle), le plafond d'accélérateur et le processeur de calcul (`inference_device: "cuda"`). Les vitesses sont **volontairement prudentes** : la priorité était zéro collision.

---

## 10. Les 10 circuits, un par un

| Circuit | Professeur | Difficulté principale | Ce qui l'a résolu |
|---|---|---|---|
| **Principal** | lignes blanches | aller vite (environ 40 s par tour) | réglage de vitesse ; données à plusieurs vitesses (sessions `fast_*` et `auto_*`) ; DART (`main_d*`) |
| **warren** | lignes blanches (`single_line`) | le professeur braquait trop peu dans le grand virage | braquage limité seulement pendant les 3 premières secondes ; 5 nouvelles sessions |
| **mini_monaco** | ligne jaune | ombre, bord de route olive, ligne de départ en damier | ligne jaune suivie en chaîne, choix du jaune le plus vif |
| **generated_road** | ligne jaune pâle | ligne pointillée, route qui se termine | `yellow_pale`, `lost_grace_s` (rouler 0,8 s sans ligne) |
| **generated_track** | CTE | **tracé différent à chaque lancement**, virages en S | sessions sur plusieurs tracés + DART ; reste le circuit le plus fragile |
| **waveshare** | poursuite | tout petit, plusieurs voies peintes ; l'ancien professeur « peinture » sortait de la route | nouvelle ligne de référence + professeur poursuite (écart divisé par 5) |
| **warehouse** | CTE | la voiture dérivait sans corriger | DART |
| **circuit_launch** | poursuite | épingle serrée : le professeur CTE tapait le mur **à chaque tour** | professeur poursuite (tourne avant l'épingle) + DART |
| **mountain_track** | CTE | côtes : il faut accélérer plus | données + pédale DART |
| **sparkfun_avc** | poursuite | portique étroit, spirale, côte, tremplin avec un quad ; 30 % du circuit jamais exploré au début | exploration automatique, ligne corrigée point par point, DART |

---

## 11. Les techniques qui ont tout changé

### 11.1 DART : apprendre à revenir sur la ligne

**DART** vient d'un article de recherche (Laskey et al., 2017). Le problème est celui de la partie 2.9 : le réseau ne voit jamais « être un peu à côté ».

La solution, codée dans `racing/drive.py` (option `baseline.perturb`) :
1. Toutes les 1,5 à 4,5 secondes environ, on **pousse le volant** de 0,3 à droite ou à gauche, pendant 0,5 s. La voiture s'écarte vraiment de sa trajectoire.
2. Mais on enregistre comme **label** ce que le professeur **voulait** faire, c'est-à-dire corriger.
3. Le réseau voit donc des images « un peu à côté », avec la bonne réponse : « tourne pour revenir ».

Les réglages de sécurité :
- `skip_nodes` : zones où l'on ne pousse jamais (portique, spirale, épingle).
- `max_teacher_steering = 0,6` : on ne pousse jamais quand le professeur tourne déjà fort. Trois sessions avaient fini contre un mur, parce que la poussée arrivait en plein virage serré.

**Effet mesuré** :

| Circuit | Sans DART | Avec DART |
|---|---|---|
| sparkfun | 0 essai réussi sur 2 | 2 sur 2 |
| warehouse | 1 sur 2 | 2 sur 2 |
| circuit_launch | 1 sur 2 | 2 sur 2 |

### 11.2 DART sur la pédale

Même idée pour la vitesse, avec les options `perturb.pedal`, `pedal_share` et `pedal_skip_nodes` :
- on coupe ou on renforce l'accélérateur pendant 1 s ;
- le label reste la pédale que le professeur voulait.

Le réseau apprend par exemple « si je suis lent dans une côte, j'accélère plus ». C'est exactement ce qui manquait sur mountain_track, où la voiture calait dans la côte.

### 11.3 Jeter la seconde avant chaque problème

Option `trim_before_invalid_s`. Si le professeur fait une erreur qui mène à une collision, les images juste avant contiennent cette erreur. On les retire.

### 11.4 Un professeur par type de circuit

Un seul professeur ne savait pas conduire partout. On a donc donné à chaque circuit le professeur qui lui convient (partie 7). **Un mauvais professeur donne un mauvais élève** : c'est ce qui s'est passé sur waveshare.

### 11.5 Tout tester dans le simulateur, plusieurs fois

Les résultats varient d'un essai à l'autre : position de départ, petits écarts de timing. Chaque modèle est donc testé **2 fois sur les 10 circuits**.

---

## 12. L'histoire du projet, version par version

| Version | Données | Nouveauté | Résultat |
|---|---|---|---|
| Professeur seul | — | professeur « lignes blanches » validé | 4 tours propres sur le circuit principal, environ 56 s par tour |
| CNN v1 | principal | premier CNN | environ 60 s par tour |
| v2 | principal | meilleurs réglages de vitesse | 42,1 s par tour |
| v3 | principal + warren | 2ᵉ circuit | 41,6 à 42,2 s ; warren pas encore fiable |
| v4 | 4 circuits (principal, warren, monaco, road) | professeurs ligne jaune, coupe avant erreur | **40,1 s** sur le principal ; 4 circuits réussis ; **échec sur les 6 autres** (jamais vus) |
| v5 | 4 circuits | **CNN + LSTM** | 10 tours sur 10 ; un peu plus rapide |
| v6 | **10 circuits** | données de tous les circuits | 8 sur 10 (ancien critère) ; circuit_launch et sparkfun échouent |
| v7 | 10 circuits | professeur **poursuite** pour circuit_launch et sparkfun | 7 sur 10 au 1ᵉʳ essai ; sparkfun dérive |
| v9 | 10 + DART sparkfun | **DART** | 17 essais réussis sur 20 |
| v10 | idem | LSTM sur 10 circuits | 10 sur 20 (moins bien que le CNN) |
| v11 | + DART circuit_launch, warehouse, principal | DART partout où ça dérivait | **19 sur 20** (critère collision) |
| v12 | + DART generated_track | | 10 sur 10 au 1ᵉʳ essai |
| — | — | **découverte** : waveshare sort de la route sans collision | nouvelle mesure stricte : v11 = 8/10 et 7/10, v12 = 8/10 |
| v13 | waveshare refait + **sortie pédale** + lots de 256 | professeur poursuite sur waveshare ; 100 % deep learning possible | mode règle : **9/10 strict** ; mode réseau : 7/10 |
| v14 | + DART pédale sur 8 circuits | le réseau apprend à gérer sa vitesse | **100 % réseau : 10/10 puis 8/10 ; règle de vitesse : 10/10** |

Les détails de chaque décision, avec les chiffres, sont dans `docs/JOURNAL_AUTONOME.md`.

---

## 13. Comment on mesure la réussite

### 13.1 Le test

Outil : `tools/final_eval.sh <modèle> <préfixe de config> 120 <étiquette>`.

Pour chacun des 10 circuits :
1. le simulateur s'ouvre ;
2. le réseau conduit 120 secondes ;
3. le simulateur se ferme.

Le résumé est écrit dans `results/final_<modèle>_<étiquette>.jsonl` : temps sans collision, tours parcourus, vitesse moyenne, temps à l'arrêt.

### 13.2 La sortie de route (que le simulateur ne voit pas)

Sur waveshare, la voiture roulait sur le logo, **en dehors de la route**, mais le simulateur ne signalait rien : il n'y a pas de mur, seulement de la peinture. On a donc ajouté `tools/offtrack_check.py` :
- **Circuits fixes** : on calcule, en mètres, la distance entre la voiture et **l'endroit le plus proche où un professeur est passé**. Plus de 30 cm = hors de la route.
- **Circuits aléatoires** (generated_road, generated_track) : la position ne veut rien dire, puisque le tracé change. On compare donc l'écart au chemin du simulateur avec celui des professeurs.
- Les 3 premières secondes (le point de départ imposé) ne comptent pas.

**Un essai est réussi** s'il n'y a **aucune collision** ET **au plus 1 seconde hors de la route**.

Cette mesure est **plus sévère** que l'ancienne. Avec elle, v11 passe de « 19 sur 20 » à « 15 sur 20 ».

---

## 14. Résultats finaux

*(Cette partie est remplie avec les derniers essais. Voir aussi `results/` et `docs/JOURNAL_AUTONOME.md`.)*

### 14.1 Le modèle final : **CNN v14**

- **Fichier du modèle** : `runs/cnn_all10_v14/best.pt`.
- **Configs de conduite** :
  - `configs/tracks/cnn_v14n_<circuit>.json` : **100 % réseau**, c'est-à-dire volant, accélérateur et frein tous par le réseau. C'est le mode qui respecte la règle « deep learning seulement ».
  - `configs/tracks/cnn_v14_<circuit>.json` : volant par le réseau, vitesse par la petite règle.
- **Données** : 64 sessions de professeurs sur les 10 circuits, dont les sessions DART (volant et pédale). Cela fait **250 035 exemples d'entraînement**, 59 227 de validation et 23 290 de test.
- **Erreur de braquage en validation : 0,0345.** Pour comparer : v13 0,0378, v12 0,042, v9 0,047.
- **Entraînement** : 60 époques en 19 minutes, carte graphique à environ 90 %.

### 14.2 Les essais dans le simulateur (critère strict)

Chaque case correspond à un essai de **2 minutes**, compté comme réussi seulement s'il n'y a **aucune collision**, **pas plus d'1 s hors de la route** et **aucune interruption**.

| Circuit | v14 100 % réseau, essai 1 | v14 100 % réseau, essai 2 | v14 avec règle de vitesse | Tours en 2 min (100 % réseau) | Vitesse moyenne |
|---|---|---|---|---|---|
| Principal | ✅ | ✅ | ✅ | 2,6 | 1,05 m/s |
| warren | ✅ | ✅ | ✅ | 2,4 | 1,04 |
| mini_monaco | ✅ | ✅ | ✅ | 1,4 | 0,8 |
| generated_road | ✅ | ✅ (voir note 1) | ✅ | route ouverte | 1,0 |
| generated_track | ✅ | ❌ bord touché à 86 s, après plus de 2 tours propres | ✅ | 4,2 | 1,0 |
| waveshare | ✅ | ✅ | ✅ | **9,3** | 0,61 |
| warehouse | ✅ | ✅ | ✅ | 2,5 | 0,95 |
| circuit_launch | ✅ | ✅ | ✅ | 2,3 | 0,94 |
| mountain_track | ✅ | ✅ | ✅ | 1,5 | 0,84 |
| sparkfun_avc | ✅ | ❌ 6 s un peu à côté (31 à 47 cm), sans collision | ✅ | 0,87 (circuit très long) | 0,97 |
| **Total** | **10/10** | **8/10** | **10/10** | | |

**Bilan.** En mode 100 % réseau, v14 a réussi **18 essais sur 20**. Avec la règle de vitesse, il a fait **10 circuits sur 10** (un passage). Aucune collision sur les 20 premiers essais en mode réseau, sauf un contact sur generated_track après 86 s.

**Note 1 (generated_road, essai 2).** Le simulateur a donné un CTE absurde, de 233 à 300. Sa ligne de référence s'est « cassée », comme dans une des sessions du professeur. Les 12 photos prises pendant l'essai montrent la voiture **parfaitement dans sa voie**. L'essai est donc jugé sur les images, et l'outil signale maintenant ce cas automatiquement.

**Note 2 (3ᵉ passage, non retenu).** Un 3ᵉ passage en mode réseau a été perturbé par l'ordinateur lui-même. Le réseau a mis 3,5 fois plus de temps que d'habitude à répondre (8,6 ms au lieu de 2,5 ms par image). Deux essais ont été arrêtés par les sécurités : une image trop vieille, puis un trou de plus de 0,5 s. Sur ce passage, il y a eu 6 réussites. Les vrais défauts vus :
- mountain_track : la voiture est restée lente dans la côte pendant environ 80 s, puis elle est repartie ;
- sparkfun : une sortie de 1,65 m pendant 5 s, sans collision.

On le signale pour être honnête. Les deux premiers passages, faits avec une machine normale, servent de résultat.

### 14.3 Comparaison de toutes les versions, avec le même critère strict

| Version | Ce qui change | Circuits propres (critère strict) |
|---|---|---|
| v9 | DART sur sparkfun | 4/10 |
| v11 | + DART sur circuit_launch, warehouse et principal | 8/10 et 7/10 |
| v12 | + DART sur generated_track | 8/10 |
| v13, règle de vitesse | nouveau professeur waveshare | 9/10 |
| v13, 100 % réseau | + sortie pédale | 7/10 |
| **v14, 100 % réseau** | **+ DART sur la pédale** | **10/10 et 8/10** |
| **v14, règle de vitesse** | | **10/10** |

Sur la seule question « y a-t-il eu une collision ? », v11 faisait déjà 19 essais sur 20. Mais la voiture sortait de la route sur waveshare sans qu'aucune collision soit comptée. C'est l'utilisateur qui l'a vu, sur une capture d'écran. **Le critère strict est plus honnête.**

### 14.4 Ce que DART sur la pédale a changé

- **mountain_track** :
  - v13 en mode réseau calait dans la côte (20 % du temps à l'arrêt, puis sortie de route) ;
  - v14 monte la côte, avec 1,5 tour propre à chaque essai normal.
- **sparkfun** : v13 touchait un objet à 119,9 s, alors que v14 a fait 2 minutes sans collision.
- Le réseau accélère un peu plus que l'ancienne règle sur plusieurs circuits :
  - circuit_launch : 0,94 m/s contre 0,75 ;
  - sparkfun : 0,97 contre 0,84 ;
  - waveshare : 0,61 contre 0,47.

  Il imite la vitesse des professeurs.


---

## 15. Ce qu'on a appris (les leçons)

1. **Les données comptent plus que le réseau.** Presque tous les progrès sont venus de meilleures données (meilleurs professeurs, DART, tous les circuits), pas d'un réseau plus gros.
2. **Le réseau copie tout, même les défauts.** Professeur qui zigzague, élève qui zigzague (waveshare). Professeur qui braque trop peu, élève qui braque trop peu (warren).
3. **Il faut apprendre à se rattraper.** Un professeur parfait ne montre jamais comment corriger une erreur. DART le montre.
4. **Une bonne note hors simulateur ne suffit pas.** Le LSTM avait la meilleure erreur de validation et conduisait moins bien.
5. **Mesurer honnêtement.** « Pas de collision » ne veut pas dire « bonne conduite ». Il faut aussi regarder si la voiture reste sur la route, et regarder les images.
6. **Généraliser à un circuit jamais vu est très difficile** avec ce type de méthode. Il faut des exemples de chaque type de décor.
7. **Ne pas pousser n'importe où.** Une perturbation DART en plein virage serré mène au mur. Il faut des zones protégées.
8. **Tester plusieurs fois.** Un seul essai peut tromper, dans les deux sens.
9. **Les détails pratiques comptent** :
   - le disque plein a fait échouer un entraînement ;
   - un redémarrage de l'éditeur arrêtait les tâches en cours (d'où les pipelines « détachés », `tools/pipeline_v13.sh` et `tools/pipeline_v14.sh`) ;
   - deux lancements du simulateur en même temps se gênent.

---

## 16. Limites et idées pour aller plus loin

**Limites :**
- **Les vitesses sont prudentes.** La voiture ne roule pas au maximum possible. Le meilleur temps sur le circuit principal (40,1 s) a été obtenu avec v4, spécialisé sur 4 circuits.
- **generated_track** reste fragile, parce que son tracé change à chaque lancement.
- Certains professeurs utilisent la position du simulateur (ils sont « privilégiés »). C'est permis, car ils ne conduisent pas pendant le test, mais il faut le dire.
- La limite de 30 cm pour la sortie de route est une approximation : on ne connaît pas exactement les bords de chaque route.

**Idées :**
1. **Aller plus vite** : enregistrer des professeurs plus rapides, circuit par circuit, puis réentraîner.
2. **Plus de tracés generated_track** : des sessions sur 10 ou 20 tracés différents.
3. **DAgger** : faire conduire le réseau, puis demander au professeur ce qu'il aurait fait dans les situations où le réseau s'est mis. C'est plus puissant que DART, mais plus long.
4. **Un humain comme professeur** sur un circuit, pour comparer.
5. **Apprentissage par renforcement**, pour améliorer le temps au tour après l'imitation.

---

## 17. Guide pratique : installer, conduire, entraîner

### 17.1 Installer (une seule fois)

Dans PowerShell, dans le dossier du projet :

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# pour la carte graphique NVIDIA, installer PyTorch CUDA (version 2.10, CUDA 12.8) :
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Les 59 tests doivent afficher `OK`. Il faut aussi **Git for Windows**, pour la commande `bash` des scripts du dossier `tools/`.

Le simulateur se trouve dans `DonkeySimWin/DonkeySimWin/donkey_sim.exe`. S'il manque, le télécharger depuis le projet officiel *gym-donkeycar / sdsandbox*, version Windows.

### 17.2 Faire conduire le modèle final

Le plus simple (le simulateur s'ouvre et se ferme tout seul) :

```bash
bash tools/sim_run.sh runs/cnn_drive cnn -- --mode cnn --checkpoint runs/cnn_all10_v14/best.pt --config configs/tracks/cnn_v14n_waveshare.json --seconds 120
```

Pour un autre circuit, change le nom du circuit à la fin du fichier de config : `cnn_v14n_warren.json`, `cnn_v14n_mini_monaco.json`, etc. Le `n` veut dire « pédale par le réseau ». Sans `n`, c'est l'ancien mode règle.

### 17.3 Tester les 10 circuits

```bash
bash tools/final_eval.sh runs/cnn_all10_v14/best.pt cnn_v14n 120 essai1
.venv/Scripts/python.exe tools/offtrack_check.py results/final_cnn_all10_v14_essai1.jsonl
```

### 17.4 Enregistrer de nouveaux exemples

```bash
SPECS="mon_groupe:expert_pedal_waveshare" bash tools/collect_path_spark.sh
```

Ce script marche pour tous les circuits, malgré son nom. Il enregistre 5 minutes avec le professeur de la config et vérifie la session. Pour vérifier une session : `python tools/check_session.py data/sessions/<id>`. Les sessions ratées vont dans `data/excluded_sessions/`.

### 17.5 Préparer et entraîner

```bash
.venv/Scripts/python.exe -m racing.data --sessions data/sessions --out data/prepared/NOUVEAU --config configs/train_v14_all.json
.venv/Scripts/python.exe -m racing.train --config configs/train_v14_all.json --dataset data/prepared/NOUVEAU --out runs/NOUVEAU --cache-dir data/cache --gpu-data --start-training
```

**Attention** :
- Les dossiers `data/prepared/...` contiennent des **chemins complets** vers les images. Après avoir copié le projet ailleurs, il faut **refaire** la préparation.
- Le cache `data/cache/*.npy` peut faire de 4 à 9 Go par entraînement. Supprime les anciens.

---

## 18. Carte des fichiers du projet

```
racing/                le programme principal (Python)
  drive.py             conduite : professeur, réseau ou manuel ; enregistrement ; DART
  baseline.py          les professeurs (lignes blanches, jaune, CTE, poursuite)
  model.py             le réseau CNN (et la variante LSTM)
  preprocess.py        découpe, réduction, normalisation des images
  data.py              préparation des exemples, labels, découpage train/val/test
  dataset.py           lecture des exemples (processeur)
  gpu_data.py          exemples et augmentation directement sur la carte graphique
  cache.py             cache des images préparées
  train.py             l'entraînement
  inference.py         chargement d'un modèle et prédiction
  control.py           du réseau aux commandes (volant, accélérateur, frein)
  transport.py, session.py, vendor/   communication avec le simulateur
  recording.py         écriture des sessions
  laps.py              détection et validation des tours
  config.py            lecture et vérification des configurations
configs/
  train_v*.json        réglages d'entraînement de chaque version
  tracks/              réglages de conduite par circuit (cnn_v14n_<circuit>.json, expert_*.json)
  paths/               lignes de référence du professeur poursuite
tools/                 scripts : test des 10 circuits, sortie de route, collecte, lignes, pipelines
tests/                 59 tests automatiques
data/sessions/         les enregistrements (images + commandes)
data/excluded_sessions/  sessions écartées (ratées ou mauvais professeur), gardées pour mémoire
runs/                  les modèles entraînés (best.pt) et les journaux de conduite
results/               résultats des tests des 10 circuits
docs/                  ce rapport, le journal détaillé, les audits
DonkeySimWin/          le simulateur
```

---

## 19. Glossaire

- **Behavioral cloning (apprentissage par imitation)** : apprendre à conduire en copiant un conducteur.
- **CNN** : réseau de neurones à convolutions, spécialisé dans les images.
- **LSTM** : réseau récurrent qui a une mémoire et lit une suite d'images dans l'ordre.
- **Label** : la bonne réponse d'un exemple (ce que le professeur a fait).
- **Perte (loss)** : mesure de l'erreur du réseau.
- **Époque** : un passage sur tous les exemples d'entraînement.
- **Batch (lot)** : paquet d'exemples traités ensemble.
- **Validation / test** : exemples jamais appris, pour mesurer honnêtement.
- **Sur-apprentissage** : apprendre par cœur au lieu de comprendre.
- **Augmentation** : petites modifications aléatoires des images.
- **Dropout** : éteindre des neurones au hasard pendant l'apprentissage.
- **CTE (cross-track error)** : écart de la voiture au chemin de référence du simulateur.
- **activeNode** : numéro du point de passage le plus proche sur le circuit.
- **Professeur (expert, baseline)** : programme qui conduit pour fabriquer les exemples.
- **Privilégié** : qui utilise une information non disponible pour le réseau, ici la position.
- **Pure pursuit (poursuite)** : viser un point un peu plus loin sur une ligne et tourner vers lui.
- **DART** : pousser la voiture de travers pendant l'enregistrement, en gardant comme label la correction du professeur.
- **Pédale** : sortie du réseau = accélérateur − frein.
- **PI (régulateur)** : règle classique qui corrige un écart, proportionnellement et en accumulant.
- **GPU / CUDA** : carte graphique et sa technologie de calcul.
- **AMP** : calcul en précision mixte, plus rapide.
- **Checkpoint (`best.pt`)** : fichier qui contient un réseau entraîné.

---

## 20. Questions fréquentes

**Le réseau voit-il la position de la voiture ?**
Non. Il ne reçoit que l'image et la vitesse. La position sert seulement à certains professeurs et aux mesures.

**Pourquoi ne pas avoir conduit nous-mêmes au clavier ?**
On pourrait : le mode `--mode manual` existe. Mais il faut des centaines de milliers d'exemples **propres** sur 10 circuits. Un professeur automatique est plus régulier et ne se fatigue pas. Et la méthode (l'imitation) reste exactement la même.

**Pourquoi la voiture ne va-t-elle pas plus vite ?**
On a d'abord visé zéro collision et zéro sortie de route sur les 10 circuits. Aller plus vite demande des professeurs plus rapides, puis un réentraînement.

**Pourquoi 3 braquages (maintenant, +0,25 s, +0,5 s) ?**
Pour que le réseau « regarde loin ». Apprendre ce qu'il faudra faire bientôt l'oblige à comprendre la forme du virage qui arrive.

**Que se passe-t-il sur un circuit jamais vu ?**
Le réseau échoue, s'il ne ressemble pas assez aux circuits appris. C'est ce qu'a montré v4 sur les 6 circuits qu'il n'avait jamais vus.

**Combien de temps faut-il pour tout refaire ?**
- Enregistrer une session : 5 minutes.
- Préparer les données : environ 10 minutes.
- Entraîner : 15 minutes.
- Tester les 10 circuits : environ 25 minutes.
