# Dents et Visage

Un blog moderne et sécurisé dédié aux soins dentaires et à la beauté du visage, avec une architecture backend sécurisée et un module ML réutilisable pour l'intégration dans des projets MLOps.

## Fonctionnalités

- **Backend FastAPI sécurisé**: Aucune "porte dérobée", suivant les meilleures pratiques de sécurité
- **Système de blog optimisé pour le SEO**: Lit les fichiers Markdown depuis `blog_content/` avec des balises meta appropriées
- **Module ML organisé**: Package Python réutilisable dans `src/alexendra/` pour l'intégration MLOps

## Structure du Projet

```
alexendra1/
├── src/
│   └── alexendra/       # Module ML réutilisable pour l'intégration MLOps
│       ├── __init__.py
│       ├── model.py
│       ├── multimodal_merger.py
│       ├── internet_crawler.py
│       ├── advanced_model.py
│       └── utils.py
├── blog_content/        # Fichiers Markdown pour les articles de blog
│   ├── dental-care-tips.md
│   └── skin-care-guide.md
├── static/              # Actifs statiques (CSS, JS, images)
│   └── style.css
├── templates/           # Modèles HTML Jinja2
│   ├── home.html
│   └── blog_post.html
├── main.py              # Point d'entrée de l'application FastAPI
├── requirements.txt     # Dépendances du projet
└── README.md
```

## Mise en Route

1. **Installer les dépendances :**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Lancer le serveur :**
   ```bash
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Visiter le site :**
   Ouvrez votre navigateur et allez sur `http://localhost:8000`

## Ajouter de Nouveaux Articles de Blog

Créez simplement un nouveau fichier Markdown (.md) dans `blog_content/`! Le nom de fichier deviendra le slug de l'URL!
