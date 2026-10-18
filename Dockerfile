# 1. Image de base officielle, version "slim" pour réduire la surface d'attaque et le poids
FROM python:3.10-slim

# 2. Métadonnées du mainteneur
LABEL maintainer="Ingénieur Data Science"
LABEL description="API de classification d'IRM (ResNet18)"

# 3. Empêche Python d'écrire des fichiers .pyc et force l'affichage direct des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Définition du répertoire de travail dans le conteneur
WORKDIR /app

# 5. Installation des librairies systèmes critiques (minimalistes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 6. Copie du fichier de dépendances EN PREMIER (Optimisation du cache Docker)
COPY requirements.txt .

# 7. Installation des dépendances Python sans garder le cache (gain d'espace disque massif)
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copie du code source et du modèle mathématique
COPY app.py .
COPY meilleur_modele_resnet_cropped.pth .

# 9. Exposition du port réseau
EXPOSE 8000

# 10. Commande de lancement du serveur asynchrone
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]