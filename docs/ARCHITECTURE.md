# Architecture du Projet

## Vue d'ensemble

```
┌─────────────┐      HTTP       ┌─────────────┐
│             │ ───────────────> │             │
│  Streamlit  │                  │   FastAPI   │
│  Frontend   │ <─────────────── │   Backend   │
│             │      JSON        │             │
└─────────────┘                  └──────┬──────┘
                                        │
                                        │ Cache
                                        v
                                   ┌─────────┐
                                   │  Redis  │
                                   └─────────┘
```

## Flux de données

1. Upload → Validation → Preprocessing
2. Mining → Cache → Sampling
3. Feedback → Reweighting → New Sample

## Technologies

- **Backend**: FastAPI, mlxtend, pandas
- **Frontend**: Streamlit, plotly
- **Cache**: Redis
- **Containerisation**: Docker
