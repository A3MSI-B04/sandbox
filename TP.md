Travaux Dirigés – Analyse de données immobilières 

Objectif général : 

À partir d’un fichier CSV (txt)  issu de la base de données publique data.gouv.fr, vous devez construire un modèle capable d’estimer la valeur d’un bien immobilier. Ce modèle devra s’appuyer sur une classification non supervisée suivie d’une régression, avec une possibilité d’aller plus loin en proposant une estimation pondérée par probabilités d’appartenance aux classes.  

 

📁 Données à utiliser : 

Vous devez choisir un fichier de données immobilières (par exemple : transactions immobilières, valeurs foncières, etc.) sur data.gouv.fr. Le fichier doit contenir des informations permettant de prédire la valeur d’un bien (prix, surface, localisation, type de bien, etc.). 

 

🧪 Travail demandé : 

1. Compréhension des données 

✅ Identifier les colonnes disponibles et leur signification. 

✅ Déterminer les variables pertinentes pour la prédiction du prix. 

Justifier vos choix dans le rapport. 

2. Nettoyage des données 

✅ Supprimer ou traiter les valeurs manquantes, aberrantes ou inutiles. 

✅ Normaliser ou encoder les variables si nécessaire. 

Documenter les étapes de nettoyage dans le rapport. 

3. Classification non supervisée 

Appliquer une méthode de clustering (ex : KMeans, DBSCAN, etc.). 

Déterminer le nombre optimal de classes. 

Interpréter les classes obtenues (quelles caractéristiques les définissent ?). 

4. Régression par classe 

Pour chaque classe identifiée, entraîner un modèle de régression (linéaire ou non linéaire). 

Évaluer les performances de chaque modèle. 

Justifier le choix des modèles utilisés. 

5. Estimation pondérée (optionnel mais valorisé) 

Pour un bien donné, calculer les probabilités d’appartenance à chaque classe. 

Estimer le prix selon chaque modèle de classe. 

Fournir une estimation finale pondérée par les probabilités. 

Exemple de sortie :  

"Ce bien appartient à la classe B avec une probabilité de 0.65, à la classe C avec 0.30, et à la classe A avec 0.05. 

 Estimation par classe B : 250 000 € 

 Estimation pondérée finale : 245 000 €" 

 

📦 Livrables attendus : 

1. Un fichier Python unique (td_immobilier.py) 

Ce fichier doit :  

Accepter un fichier CSV en entrée. 

Contenir tout le pipeline : nettoyage, clustering, régression. 

Permettre de prédire le prix d’un bien donné (via une ligne du CSV ou un dictionnaire). 

Afficher les résultats de manière claire. 

2. Un rapport PDF 

Présentation de la démarche suivie. 

Réponses aux questions posées. 

Justification des choix techniques. 

Analyse des résultats obtenus. 

Discussion sur les limites et les améliorations possibles. 

 

🧠 Conseils : 

Utilisez des bibliothèques comme pandas, scikit-learn, matplotlib, seaborn, etc. 

Pensez à structurer votre code avec des fonctions claires. 

Documentez votre code pour faciliter la lecture. 

Testez votre modèle sur plusieurs exemples. 

 

🏆 Critères d’évaluation : 

Critère 

Points 

Nettoyage et préparation des données 

4 

Qualité de la classification 

4 

Qualité des modèles de régression 

4 

Estimation pondérée (bonus) 

+2 

Clarté du code et du rapport 

4 

Pertinence des analyses et interprétations 

4 

 