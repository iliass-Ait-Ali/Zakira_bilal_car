# Vérifications réellement effectuées

Date : 15 novembre 2026.

**Ce rapport décrit la préparation initiale.** Les interventions ultérieures ont validé [la connexion réelle](CONNEXION_SDSANDBOX.md), puis [quatre tours autonomes](TOURS_VALIDES.md). La suite atteint maintenant 42 tests réussis. Les mentions « non exécuté » ci-dessous concernent uniquement l'étape initiale.

| Vérification | Résultat observé |
|---|---|
| Lecture complète des 336 images historiques | 336 RGB 160×120, aucune corrompue, aucun doublon de fichier, aucun label associé |
| Inspection des scripts et dépendances | Audit enregistré dans `AUDIT.md` |
| Inspection statique des deux DLL Unity | Instructions IL et hashes dans `binary_protocol_il.txt` ; aucune exécution du binaire |
| `python -m racing.check` | Python 3.13.4 ; torch 2.10.0, NumPy 2.2.6, Pillow 11.2.1, OpenCV 4.13.0.90, Matplotlib 3.10.3 ; CUDA non disponible |
| Passage avant synthétique | 232 751 paramètres ; sortie `[1,3]`, valeurs finies ; aucune mesure de latence de course |
| `python -m unittest discover -s tests -v` | **20 tests réussis** |
| `python -m compileall -q racing tests tools` | Terminé sans erreur |

Les tests couvrent notamment les dimensions à une et trois images, RGB et normalisation, inversion des labels lors du flip, absence d'augmentation en validation, masques des horizons futurs, séparation des groupes, gestion des images corrompues et des ruptures, enregistrement/chargement de sessions synthétiques, pertes masquées, bornes des commandes, freinage anticipé, limitation de pente, arrêt verrouillé et watchdog sans socket réelle. Une sérialisation de poids aléatoires explicitement **non entraînés** est testée dans un dossier temporaire puis supprimée.

Les premiers essais de tests avaient échoué sur les permissions des dossiers temporaires Windows, sans erreur algorithmique identifiée dans ces cas. Les fixtures utilisent désormais un répertoire temporaire dans le workspace, avec contrôle de son chemin avant nettoyage. La suite finale ci-dessus passe.

## Non exécuté / non démontré

- Aucune connexion au port 9091, aucun lancement du simulateur, aucune commande de conduite réellement envoyée.
- Aucun appel à un optimiseur, aucun backward, aucun entraînement, fine-tuning ou recherche d'hyperparamètres.
- Aucun checkpoint entraîné créé ; aucun résultat de conduite ou de compétition inventé.
- Aucune collecte réelle, aucun calcul de métriques sur un modèle entraîné, aucun benchmark de latence.
- Aucun essai GPU ou réinstallation complète des dépendances dans un environnement vierge. Les versions épinglées correspondent à l'environnement local vérifié ; le README donne la création d'un environnement isolé et `pip check` à exécuter.
- L'entraînement complet, le transport en conditions réelles, les paramètres physiques, la disponibilité des champs télémétriques en compétition et les performances restent à valider aux étapes autorisées ultérieurement.

Les téléchargements réseau effectués pendant l'audit concernent uniquement les sources publiques de référence et deux petits lecteurs statiques .NET (`dnfile`, `dncil`), sans charger de poids préentraînés ni interroger le simulateur.
