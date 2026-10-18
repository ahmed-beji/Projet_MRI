# Projet_MRI
# 🧠 Pipeline MLOps & Deep Learning : Classification de Tumeurs Cérébrales (IRM)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=MLflow&logoColor=white)

##  Contexte Clinique et Objectif d'Ingénierie
La classification automatique des tumeurs cérébrales par IRM est souvent biaisée par des métriques globales trompeuses (Accuracy). Ce projet vise à construire un système d'aide au diagnostic robuste de niveau industriel, capable de classifier 4 pathologies (Gliome, Méningiome, Tumeur de l'hypophyse, Cerveau sain) tout en **minimisant mathématiquement le risque de faux négatifs critiques** (ex: renvoyer un patient atteint d'un gliome avec un diagnostic sain).

##  Architecture du Pipeline

Ce projet dépasse le simple prototypage en Jupyter Notebook et intègre les standards MLOps de production :

### 1. Ingénierie de la Donnée & Traitement des Biais (OpenCV)
Un audit initial via **Grad-CAM** a révélé que le modèle développait un biais spatial (raccourci mathématique focalisé sur la boîte crânienne). 
- **Solution :** Implémentation d'un algorithme d'Auto-Cropping (Skull Stripping) via `cv2.findContours` et seuillage morphologique pour forcer l'extraction de caractéristiques sur le tissu cérébral (Parenchyme) uniquement.

### 2. Modélisation & Optimisation (PyTorch)
- **Architecture :** Transfer Learning sur ResNet18.
- **Fine-Tuning Différentiel :** Gel des blocs convolutifs superficiels (`layer1` à `layer3`) et déblocage profond (`layer4`) avec un taux d'apprentissage microscopique ($10^{-6}$) pour éviter le *catastrophic forgetting*.
- **Loss Pondérée :** Implémentation d'une `CrossEntropyLoss` asymétrique (poids x3 sur les gliomes) pour contraindre mathématiquement le modèle à sur-sécuriser la classe la plus difficile.
- **Scheduler :** Intégration d'un `ReduceLROnPlateau` pour stabiliser la descente de gradient et empêcher l'overfitting.

### 3. MLOps, Déploiement & Inférence
- **MLflow :** Traçabilité totale des hyperparamètres, versioning du fichier `.pth` et suivi des courbes de Loss/Accuracy.
- **FastAPI :** Conteneurisation de la fonction d'inférence dans une API REST asynchrone à haute performance.
- **Docker :** Fichier `Dockerfile` industriel multi-stage, optimisé spatialement (PyTorch CPU-only & OpenCV Headless).

## Performances Cliniques
Le modèle optimisé a atteint une **Accuracy globale de 93%**, avec un comportement de sauvegarde clinique validé :
- **Rappel (Recall) sur le Méningiome :** 94%
- **Faux négatifs critiques (Gliome confondu avec Cerveau sain) :** Réduction drastique confirmée par la matrice de confusion.
- L'audit final Grad-CAM confirme que l'attention des gradients $\nabla_{\theta}$ est strictement localisée sur la nécrose tumorale.

##  Installation et Déploiement (Docker)

Le modèle est encapsulé et prêt à être déployé sur n'importe quelle infrastructure locale ou Cloud.

1. **Cloner le dépôt :**
```bash
git clone (https://github.com/ahmed-beji/Projet_MRI.git)
cd projet_MRI


├── app.py                            # Serveur FastAPI (Route /predict et Auto-Crop OpenCV)
├── Dockerfile                        # Instructions de construction du conteneur optimisé
├── requirements.txt                  # Dépendances strictes (PyTorch CPU, OpenCV Headless)
├── meilleur_modele_resnet_cropped.pth # Poids optimisés et versionnés du modèle ResNet18
├── Notebook_Entrainement.ipynb       # Code source de la recherche, Fine-Tuning et MLflow
└── README.md