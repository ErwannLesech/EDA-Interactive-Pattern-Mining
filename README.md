# 🔍 Pattern Mining Interactive

Projet EDA - EPITA SCIA-G  
Fouille interactive de motifs avec préférences utilisateur

## 📋 Description

Application web interactive permettant l'extraction, l'échantillonnage et la visualisation de motifs fréquents dans des données transactionnelles.

## 🏗️ Architecture

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Containerisation**: Docker Compose

## 🚀 Installation Rapide

### Avec Docker (recommandé)

```bash
docker-compose up --build
```

Accès :
- Frontend : http://localhost:8501
- Backend API : http://localhost:8000
- Documentation API : http://localhost:8000/docs

### Sans Docker

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Utilisation

1. Uploader un fichier CSV/Excel contenant des transactions
2. Configurer les paramètres d'extraction (support, confidence)
3. Visualiser les motifs découverts
4. Utiliser le feedback (👍/👎) pour affiner les résultats
5. Exporter les motifs sélectionnés

## 📁 Structure du Projet

```
pattern-mining-interactive/
├── backend/          # API FastAPI
├── frontend/         # Interface Streamlit
├── data/            # Datasets d'exemple
├── tests/           # Tests unitaires
└── docs/            # Documentation
```

## 👥 Équipe

- Lesech Erwann
- Le Riboter Aymeric
- Aubron Abel
- Claude Nathan

## 📝 License

MIT License - EPITA 2025
