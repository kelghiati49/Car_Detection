🔹 Stratégie :
-L'utilisateur upload une vidéo via Streamlit.
-La vidéo est envoyée directement à FastAPI en une seule requête.
-FastAPI traite la vidéo et renvoie le lien vers la video apres son traitement intitule "output_detected.
-Streamlit affiche l'output.

🔹 Notebook :
- Car_detect.py : contient les modèles d'IA utilisés pour la détection
  et le suivi des véhicules.
- backend.py : utilise FastAPI pour faire le lien entre le frontend et Car_detect.py.
- frontend.py : utilise Streamlit pour créer l'interface graphique du projet.

🔹 Dépendances :
- Nous travaillons sur une VM créée via Anaconda, avec Python 3.9.21.
Les versions des bibliothèques utilisées dans ce projet se trouvent
dans requirements.txt.

🔹 Limites :
- Notre modèle d'IA (Car_detect.py) ne s'adapte pas bien à tous les types de vidéos.
Pour cette raison, nous utilisons la vidéo intitulée "CarInWay", qui se trouve
dans le même dossier que le projet.
Nous allons travailler avec cette vidéo, puis nous procéderons à la résolution de cette limitepar la suite.
