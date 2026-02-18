FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# deps système (souvent nécessaires selon tes libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Installation dépendances
# Si tu as pyproject.toml:
COPY pyproject.toml ./
COPY README* ./
# Si tu as requirements.txt, utilise plutôt ça (voir note plus bas)
# COPY requirements.txt ./

# Copie du code
COPY . .

# Install python package
RUN pip install --upgrade pip && pip install -e .

EXPOSE 8501
CMD ["streamlit", "run", "canar/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
