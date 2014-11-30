Projet-2A-Info-GPU
==================

Instructions :

 - `make` : Construit la bibliothèque images et les exécutables (GPP / Tests).
 - `make run_gpp` : Construit les images de sortie de chaque algorithme implémentés sur GPP.
 - `make run_tests_gpp` : Lance les tests de validité sur les images de sortie des différents algorithmes implémentés sur GPP.
 - `make clean` : Nettoie les fichiers temporaires.
 - `make cleanall` : Nettoie tous les fichiers construits lors de la commande make. 

Les images de sortie générées se trouvent dans gpp/output-images.
Les images de référence se trouvent dans references/output-images.

Les nombres renvoyées par la commande "make run_gpp" représentent :
- Le nombre de cycles d'horloge lors de l'exécution de l'algorithme.
- Le temps réel en seconde de l'exécution de l'algorithme.