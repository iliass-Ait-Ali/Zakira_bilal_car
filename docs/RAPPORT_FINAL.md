# Rapport final : voiture autonome par deep learning (DonkeyCar / SDSandbox)

> **Version courte, arrêtée à v11.** Pour le rapport complet et à jour (modèle final v14, 100 % réseau, mesure stricte avec sortie de route), lire **[RAPPORT_COMPLET.md](RAPPORT_COMPLET.md)**.

## 1. Le but

Une voiture du simulateur DonkeyCar doit rouler **toute seule**, **avec la caméra seulement**, sur les **10 circuits** du simulateur.

On dit qu'un circuit « marche » si la voiture roule **2 minutes, ou fait 1 tour complet, sans aucune collision**.

## 2. Comment ça marche (en résumé)

```
 caméra 255×255 ──► découpe + réduction 160×96 ──► CNN ──► braquage prévu maintenant, dans 0,25 s et dans 0,5 s
 vitesse ────────────────────────────────────────┘                    │
                                                                      ▼
                          régulateur (vitesse selon le virage, frein) ──► volant / accélérateur / frein
```

- **Apprentissage par imitation** (« behavioral cloning ») : un professeur automatique conduit, on enregistre chaque image avec le braquage qu'il a choisi, et le réseau apprend à reproduire ce choix à partir de l'image seule.
- **Le réseau CNN** : 5 couches de convolution (24, 36, 48, 64, 64 filtres), puis 2 couches denses (100 et 50 neurones), environ 233 000 paramètres. Il reçoit aussi la vitesse. Il prédit 3 braquages : maintenant, dans 0,25 s et dans 0,5 s.
- **La variante CNN + LSTM** (réseau récurrent, option `model.temporal = "lstm"`) : le même CNN regarde les 5 dernières images (une toutes les 0,1 s), puis un LSTM de 128 neurones les lit dans l'ordre. Le modèle « voit » donc le mouvement. Environ 708 000 paramètres.
- **La carte graphique** (RTX 4060) : les images sont préparées une seule fois puis gardées dans la mémoire de la carte, et les variations de luminosité, de contraste et de miroir sont faites sur la carte. Une époque prend environ 1 minute.

## 3. Les professeurs (pour fabriquer les données)

Il n'y a **aucune donnée manuelle** : chaque circuit a un professeur automatique.

| Professeur | Comment il conduit | Circuits |
|---|---|---|
| Lignes blanches | suit les deux lignes blanches vues par la caméra | principal (roboracingleague_1), warren |
| Ligne jaune | suit la ligne jaune centrale, en chaîne, de bas en haut | mini_monaco, generated_road |
| Peinture | suit la ligne peinte la plus proche de chaque côté | waveshare |
| CTE | suit la distance à la ligne de référence du simulateur | generated_track, warehouse, mountain_track |
| Poursuite (`path`) | vise un point 0,8 m plus loin sur une ligne enregistrée | circuit_launch, sparkfun_avc |

⚠️ **Point important et honnête.** Les professeurs CTE et Poursuite ont un **accès privilégié** : ils connaissent la position de la voiture donnée par le simulateur.
- Ce sont seulement des **professeurs**. Le **réseau entraîné**, lui, apprend et conduit **uniquement avec la caméra et la vitesse**.
- Si le règlement du projet impose un professeur qui ne regarde que l'image, il faudra présenter ces 5 circuits à part.

Quelques techniques utilisées pour obtenir des données propres :
- On **jette la dernière seconde avant chaque problème** (collision, ligne perdue), pour ne pas apprendre les erreurs.
- **Injection de bruit (DART)** sur sparkfun : de temps en temps, on pousse le volant à côté pendant 0,5 s, mais on enregistre ce que le professeur *voulait* faire. Le réseau apprend ainsi à **revenir** sur la trajectoire quand il s'en écarte.
- Chaque session sert en entier à **un seul** ensemble (entraînement, validation ou test). On ne teste donc jamais sur une session déjà apprise.

## 4. Résultats

Test dans le simulateur : 2 minutes par circuit, conduite **avec la caméra seulement**, **2 essais par circuit**. Critère strict : 120 secondes sans **aucune** collision.

| Circuit | v7 | v9 (+ DART sparkfun) | v10 (LSTM) | **v11 (+ DART sur 3 circuits)** | Tours en 2 min (v11) |
|---|---|---|---|---|---|
| Principal (roboracingleague_1) | ✅ | 1/2 | 0/2 | **2/2** | 3,0 |
| warren | ✅ | 2/2 | 2/2 | **2/2** | 2,5 |
| mini_monaco | ❌ | 2/2 | 2/2 | **2/2** | 1,6 |
| generated_road | ✅ | 2/2 | 2/2 | **2/2** | route ouverte |
| generated_track | ✅ | 2/2 | 0/2 | **1/2** | 3,5 |
| waveshare | ✅ | 2/2 | 2/2 | **2/2** | 6,7 |
| warehouse | ❌ | 1/2 | 0/2 | **2/2** | 2,4 |
| circuit_launch | ✅ | 1/2 | 1/2 | **2/2** | 1,9 |
| mountain_track | ✅ | 2/2 | 1/2 | **2/2** | 1,5 |
| sparkfun_avc | ❌ | 2/2 | 0/2 | **2/2** | 0,75 (long circuit) |
| **Essais réussis** | 7/10 | 17/20 | 10/20 | **19/20** | |

(v7 n'a eu qu'un essai par circuit.)

Ce qu'on retient :
- **L'injection de bruit (DART) est ce qui a le plus aidé.** Sans elle, le réseau ne savait pas revenir sur la ligne quand il s'en écartait un peu. Avec elle, sparkfun passe de 0 à 2 essais réussis sur 2, puis warehouse et circuit_launch de 1/2 à 2/2.
- **Le LSTM** (v10) a une meilleure erreur de validation (0,037 contre 0,047), mais il **conduit moins bien**. C'est un bon exemple : une bonne note sur les données enregistrées ne garantit pas une bonne conduite. Le modèle retenu est donc le **CNN v11** ([runs/cnn_all10_v11/best.pt](../runs/cnn_all10_v11/best.pt)).
- Le seul échec de v11 a lieu sur generated_track, un circuit **tiré au hasard à chaque lancement**.

Fichiers des résultats : `results/final_cnn_all10_v*_r*.jsonl`.

## 5. Ce qui a été le plus difficile

1. **circuit_launch** : le professeur CTE touchait le mur dans l'épingle **à chaque tour**. La seconde avant chaque choc étant jetée, le réseau n'avait jamais vu cette épingle.
   Solution : le professeur Poursuite, qui commence à tourner avant l'épingle. Résultat : 18 tours sans aucun choc.
2. **sparkfun_avc** : un portique étroit, une spirale, une côte, un tremplin avec un quad à côté. 30 % du circuit n'avait jamais été parcouru.
   Solution : exploration automatique, puis ligne de référence corrigée point par point, puis injection de bruit.
3. **Généraliser à un circuit jamais vu** : le réseau entraîné sur 4 circuits échouait sur tous les autres. **Il faut des données de chaque type de circuit.**

## 6. Limites connues

- Les résultats varient d'un essai à l'autre (position de départ aléatoire, et circuits régénérés à chaque lancement pour generated_track et generated_road). On a donc fait plusieurs essais par circuit.
- Les vitesses sont prudentes sur les circuits difficiles.
- Pour le circuit principal, le meilleur temps rapide (40,1 s par tour) a été obtenu avec le modèle v4, spécialisé sur 4 circuits.

## 7. Refaire les résultats

```bash
# préparer les données et entraîner (carte graphique)
.venv/Scripts/python.exe -m racing.data --sessions data/sessions --out data/prepared/all10_v4 --config configs/train_v11_all.json
.venv/Scripts/python.exe -m racing.train --config configs/train_v11_all.json --dataset data/prepared/all10_v4 --out runs/cnn_all10_v11 --cache-dir data/cache --gpu-data --start-training

# tester les 10 circuits (le simulateur s'ouvre et se ferme 10 fois)
bash tools/final_eval.sh runs/cnn_all10_v11/best.pt cnn_v11 120
```
