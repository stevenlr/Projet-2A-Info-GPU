Projet-2A-Info-GPU
==================

Instructions :

 - `make` : Construit la bibliothèque images et les exécutables (GPP, GPP-SSE (requiert les extensions SSE, SSE2 et SSSE3) et tests).
 - `make run_gpp` : Construit les images de sortie de chaque algorithme implémenté sur GPP.
 - `make run_tests_gpp` : Lance les tests de validité sur les images de sortie des différents algorithmes implémentés sur GPP.
 - `make run_sse` : Construit les images de sortie de chaque algorithme implémenté sur GPP avec extension SSE.
 - `make run_tests_sse` : Lance les tests de validité sur les images de sortie des différents algorithmes implémentés sur GPP avec extension SSE.
 - `make clean` : Nettoie les fichiers temporaires.
 - `make cleanall` : Nettoie tous les fichiers construits lors de la commande make. 

Les images de sortie générées se trouvent dans `gpp/output-images` et `gpp-sse/output-images`.  
Les images de référence se trouvent dans `references/output-images`.

Les nombres renvoyés par les commandes `make run_gpp` et `make run_sse` représentent :
- Le nombre de cycles d'horloge lors de l'exécution de l'algorithme.
- Le temps réel en seconde de l'exécution de l'algorithme.