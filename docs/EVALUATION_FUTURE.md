# Étape après votre entraînement

Ces commandes préparent l'évaluation future du CNN, qui n'a pas été entraîné ni évalué. Après la réparation de connexion, [quatre tours du pilote déterministe ont été validés](TOURS_VALIDES.md), dont trois consécutifs en 180 s. Commencer l'évaluation CNN seulement après la vérification de `best.pt` et l'analyse de ses courbes.

## Contrôles avant les premiers tours

1. Vérifier caméra, signe de direction et freinage du véhicule actif lors d'une courte conduite manuelle. Le chemin `UnityStandardCarAdapter` présente une incompatibilité potentielle du frein positif : si le véhicule actif l'utilise, il faut résoudre cette configuration côté simulateur, pas contourner le bornage par une commande négative.
2. Vérifier `image`, `time`, `speed`, résolution, fréquence reçue et horodatages de décision. L'IMU, le CTE et la position ne sont pas requis pour le pilote.
3. Garder une copie du checkpoint, de la configuration, du dataset et de l'empreinte du simulateur. Le script de conduite ne charge pas un checkpoint marqué non entraîné.
4. Définir les règles de réussite d'un tour et le nombre d'essais **avant** les essais. Le règlement officiel reste à fournir, notamment capteurs autorisés, sorties pénalisées, remises à zéro et calcul des points. Si le champ vitesse n'est pas autorisé, cette architecture devra être adaptée et réentraînée ; ne pas le masquer par une constante.

## Évaluation hors ligne

```powershell
.\.venv\Scripts\python.exe -m racing.evaluate offline --checkpoint runs/cnn_v1/best.pt --dataset data/prepared/v1 --out runs/cnn_v1/test_metrics.json
```

Le test final mesure la MAE par horizon, en ligne droite, en virage et en récupération. Ne l'utiliser qu'une fois le modèle choisi sur la validation. Les erreurs d'imitation ne mesurent pas la réussite en boucle fermée ; un modèle légèrement moins bon hors ligne peut mieux conduire et inversement.

## Pilotage et journaux

Lancer soi-même SDSandbox, puis :

```powershell
.\.venv\Scripts\python.exe -m racing.drive --mode cnn --connect --checkpoint runs/cnn_v1/best.pt --display --seconds 120
```

Avec `--display`, maintenir Espace permet une reprise manuelle. Une reprise invalide l'essai autonome concerné ; elle ne doit pas être omise des statistiques. Depuis la correction de connexion, le compteur de secondes commence après réception de la première image conforme et activation de la conduite ; pour comparer, mesurer les tours dans le temps du simulateur.

Chaque run obtient un dossier unique sous `runs/driving`, contenant la configuration, le hash du checkpoint, les décisions, les événements et la raison d'arrêt. La latence `inference_ms` inclut prétraitement, réseau et transfert de la sortie. `decision_age_s` mesure réception→envoi, pas capture→actionneur. Les trames sautées sont visibles par les identifiants `frame_id` ; conserver ces traces pour comparer CPU et GPU.

Pilote de référence réintégré, avec saturation réseau explicite et frein de protection lorsque les bords disparaissent :

```powershell
.\.venv\Scripts\python.exe -m racing.drive --mode baseline --connect --display --seconds 120
```

Il utilise désormais des bandes blanches fines, la médiane des centres et une régulation de vitesse prudente. Cet expert a passé quatre tours sur la piste et la caméra documentées ; l'ancien script et `legacy_baseline` sont conservés pour comparaison. Utiliser les mêmes configurations et conditions pour une comparaison future avec le CNN.

La collecte utilise exclusivement `--mode baseline --connect --record --baseline-validated --group reference_01`. Ajouter `--record-limit 40` pour un contrôle borné. Depuis l'audit du dataset, `data.allowed_sources` vaut uniquement `["baseline"]` et les sources manuelles/CNN sont refusées dans tous les splits. Voir [le bilan du dataset](DATASET_BASELINE.md).

## Tours, sorties et résultats

Le script reçoit les croisements de ligne mais ne transforme pas automatiquement deux croisements en tour homologué. Les candidats sont rapportés séparément. Les collisions ne détectent pas toutes les sorties ; le CTE peut manquer et ses unités/son seuil doivent être calibrés. Sans preuve, `success_rate`, temps homologués, sorties et score officiel restent `null`.

Après revue, créer un JSON d'annotations avec une entrée par **tentative**, y compris celles interrompues. Structure, à remplir avec des valeurs réelles :

```json
{
  "attempts": [
    {"success": false, "lap_seconds": null, "reason": "à remplacer par le motif réellement observé"}
  ]
}
```

Pour une tentative réussie, utiliser `"success": true` et sa durée réelle positive dans `lap_seconds`. Ne pas compter un tour comportant une reprise manuelle comme réussite autonome. Documenter les sorties, collisions et conditions particulières dans `reason`. Le champ facultatif `offtrack_events` est un entier positif ou nul : s'il est renseigné pour toutes les tentatives, le rapport calcule aussi le total des sorties revues manuellement. Aucun fichier d'annotations réel n'a été créé pendant la préparation.

```powershell
.\.venv\Scripts\python.exe -m racing.evaluate summarize --logdir runs/driving/IDENTIFIANT --annotations annotations_reelles.json --out runs/bilan_cnn.json
.\.venv\Scripts\python.exe -m racing.compare runs/bilan_baseline.json runs/bilan_cnn.json --out runs/comparaison.json
```

Les résultats incluent vitesse moyenne pondérée par le temps, vitesse maximale, variation de direction par seconde, latence médiane/p95, délai de décision p95, collisions et indicateurs disponibles. Les lacunes de journalisation sont exclues de l'intégration temporelle ; le CTE indique sa couverture. Le comparateur présente les compromis réussite/temps, sans fabriquer une formule de score ni une preuve statistique de supériorité.

Quand le barème sera fourni, implémenter une fonction `score(report)` dans un module Python puis passer `--score-plugin module:score` à `summarize`. La fonction doit gérer explicitement les métriques manquantes et ne s'exécute que si elle est demandée.

## Comparaison équitable

- Même binaire, circuit, véhicule, caméra, résolution, physique, machine et charge système ; sauvegarder chaque `run.json`.
- Même position initiale et procédure de départ, même sens de course, même fenêtre d'évaluation, mêmes scénarios et mêmes règles d'interruption.
- D'abord comparer à allure modérée comparable, puis comparer chaque pilote avec son réglage optimisé selon le même budget d'essais.
- Prévoir par exemple 10 tentatives par configuration finaliste, réparties en runs indépendants ; 10 est un plan initial, pas une taille statistiquement suffisante garantie.
- Alterner l'ordre baseline/CNN pour limiter les effets d'ordre. Inclure les échecs, les délais dépassés et les interventions humaines dans le dénominateur ; rapporter séparément les échecs de connexion avant le départ.
- Reporter meilleur temps et moyenne uniquement sur tours homologués, avec taux de réussite et nombre de tentatives. Un meilleur temps obtenu une fois sur dix n'est pas équivalent à neuf réussites sur dix.
- Faire les réglages sur des essais de développement ; conserver des runs/scénarios de validation finale non utilisés pour le tuning.
- Tester ensuite des variations autorisées : point de départ, décentrage, allure, éclairage. Ne pas modifier clandestinement la physique ou utiliser des informations interdites par la compétition.

## Optimisation progressive

Générer des propositions, sans exécuter d'expérience :

```powershell
.\.venv\Scripts\python.exe -m racing.experiments --out configs/proposals
```

Commencer par les erreurs de direction et la qualité des données. En cas de sous-apprentissage, examiner LR, capacité et couverture. En cas d'overfitting, améliorer diversité, dropout et weight decay. Comparer ensuite une image à trois images avec exactement les mêmes groupes, et mesurer les abandons de fenêtres causés par les trous temporels.

Une fois les tours fiables, relever la vitesse de ligne droite par petits paliers, puis celle des virages. Une entrée de virage trop rapide peut appeler davantage d'anticipation ou un meilleur freinage ; un retard de direction peut appeler moins de filtrage, pas seulement une vitesse réduite. Régler la réaccélération et les gains PI à partir des courbes vitesse/consigne/throttle/brake. Ne jamais ajuster tous les paramètres simultanément.

Après calibration des unités, du véhicule et de la décélération, une version future pourra convertir les horizons en distance et utiliser une enveloppe de freinage. La formule d'accélération latérale exige une courbure géométrique et une adhérence validées ; la V1 ne les invente pas.

## Amélioration par nouvelles démonstrations automatiques

Le projet utilise exclusivement l'expert déterministe. Une nouvelle collecte peut varier prudemment les consignes de vitesse, puis être auditée avant inclusion :

```powershell
powershell -NoProfile -File tools/collect_baseline.ps1 -FirstGroup 6
```

Ce script prépare cinq exécutions autonomes de 300 s, avec des consignes prudentes de 0,85 à 1,00. Les variantes plus lentes n'ont pas été testées pendant l'audit ; le script vérifie au moins trois tours par exécution et s'arrête sur échec. Ni les actions humaines ni les décisions du CNN ne deviennent des labels d'entraînement. Les récupérations depuis des écarts importants ne sont pas encore couvertes par ces profils.

Ajouter ces groupes au dataset, conserver les démonstrations initiales, préparer une nouvelle version puis entraîner un nouveau run. Garder le test final initial hors des décisions de tuning. L'apprentissage par renforcement est différé : il demanderait une fonction de récompense compatible avec les points officiels, un budget d'interactions et une procédure de validation robustes qui ne sont pas disponibles aujourd'hui.

## Intégration DonkeyCar

`DonkeyCNNPart` dans `racing/inference.py` expose `run(image_rgb, speed)` et retourne `(steering, throttle, brake)` sans socket. Dans un véhicule DonkeyCar existant, relier ces trois sorties à la partie de simulation, dont `run_threaded` accepte aussi le frein. Fournir une image RGB uint8 fraîche, la vitesse du même cycle et une surveillance des délais au niveau du véhicule. Un composant synchrone ne peut pas protéger seul un processus entièrement bloqué.

Le client TCP autonome reste l'entrée de référence de cette préparation. L'intégration à `manage.py` n'a pas été exécutée ni annoncée compatible sans essais ; les interfaces sont documentées et le dépôt DonkeyCar est préservé.
