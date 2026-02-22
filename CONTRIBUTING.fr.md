[🇫🇷 Français](CONTRIBUTING.fr.md) | [🇬🇧 English](CONTRIBUTING.md)

# Contribuer à CanaR

Merci de prendre le temps de contribuer ! Ce document décrit le workflow que nous utilisons pour garder des changements faciles à relire et reproductibles.

## Code de conduite

Merci de rester respectueux·se et constructif·ve. (Un fichier `CODE_OF_CONDUCT.md` dédié pourra être ajouté plus tard.)

## Bien démarrer

Si vous développez sur Windows, préférez suivre le guide [Développement sous Windows](#développement-sous-windows-wsl2--docker-desktop)

### 1) Fork et clone (contributeurs externes)

1. Forkez le dépôt sur GitHub
2. Clonez votre fork :
   ```bash
   git clone https://github.com/<votre-utilisateur>/canar.git
   cd canar
   ```
3. Ajoutez le remote upstream :
   ```bash
   git remote add upstream https://github.com/Romanovytch/canar.git
   git fetch upstream
   ```

Les mainteneurs peuvent travailler directement sur le dépôt principal.

### 2) Environnement local

#### Pré-requis
- Python >= 3.10
- Docker + Docker Compose (ou Docker Desktop sur Windows)
- (Optionnel) Make

Pour les utilisateurs de Windows, le 

#### Démarrer les dépendances locales (Qdrant + Postgres)
```bash
cd infra
docker compose up -d
docker compose ps
```

ou `make up` si vous utilisez le Makefile.

UI Qdrant : `http://localhost:6333/dashboard`

#### Créer un environnement virtuel
```bash
cd ..
python -m venv .venv
source .venv/bin/activate
```

ou `make venv` si vous utilisez le Makefile (ne faire qu'une seule fois pour créer l'environnement virtuel).

#### Installer CanaR (mode editable)
```bash
pip install -U pip
pip install -e ".[dev]"
```

> `pip install -e ".[dev]"` installe des outils essentiels au développement, comme **pytest** pour les tests et **ruff** pour le formattage. 

ou `make install` si vous utilisez le Makefile.

### 3) Configuration

Créez un fichier `.env` à la racine du projet à partir de `.env.example`.

> Note : `QDRANT_URL` et `DB_POSTGRES_URL` dépendent de l’endroit où CanaR tourne.
> - CanaR lancé sur l’hôte (venv) : utiliser `localhost`
> - CanaR lancé dans Docker : utiliser les noms de services Docker `qdrant`, `postgres`

## Workflow

### 1) Créer ou choisir une issue

Nous suivons le travail via GitHub Issues et des boards :
- Board CanaR : https://github.com/users/Romanovytch/projects/2
- Board AgoRa : https://github.com/users/Romanovytch/projects/1

Quand vous ouvrez une issue, merci de :
- Ajouter des labels si possible (au moins un `area:*`, un `type:*`, un `priority:*`)
- Décrire le problème et le résultat attendu
- Fournir des critères d’acceptation et des dépendances si nécessaire

### 2) Créer une branche

Créez une branche à partir de la branche de développement courante (par ex. `dev-*` si utilisé) ou de `main` si le dépôt n’utilise qu’une branche stable.

Convention de nommage :
- `<type>/<numeroIssue>-<slug-court>`

Exemples :
- `feat/123-add-default-agent`
- `fix/45-qdrant-timeout`
- `docs/77-update-readme`

### 3) Messages de commit

Utilisez des messages clairs et orientés action. Si pertinent, référencez le numéro d’issue :
- `Fix retrieval ranking for multi-collection queries (#123)`
- `Docs: clarify docker hostnames (qdrant/postgres) (#77)`

### 4) Lancer les tests, l'analyse du code et le formattage localement

#### Tests

Avant d’ouvrir une PR :
```bash
pytest
```
ou `make test`

#### Analyse & Formattage

Ruff permets d'analyser la qualité du code pour de potentielles erreurs (aussi appelé *lint*) :
```bash
ruff check .
```
ou `make lint`

Vérifier si les fichiers sont bien formattés :
```bash
ruff format --check
```
ou `make format-check`

> Pour lancer un formattage automatique :
> ```bash
> ruff check . --fix
> ```
> ou `make format`

Si vous utilisez le makefile, `make ci` lance les tests, l'analyse du code et la vérification du formattage.

### 5) Ouvrir une Pull Request

Ouvrez une PR vers la branche de développement principale (souvent `main`, ou `dev-*` si précisé dans le dépôt).

Bonnes pratiques :
- Lier l’issue : `Closes #123`
- Résumer les changements
- Indiquer “How to test” (comment tester)
- Mettre à jour la doc si le comportement / la config change

### 6) Revue et merge

- Contributeurs externes : au moins **1 approbation** est requise (protection de branche).
- Les checks CI doivent passer avant merge (quand activés).
- Préférer **Squash and merge** pour garder un historique propre (linéaire).

## Conseils de développement

### Débogage
- Logs Docker :
  ```bash
  cd infra
  docker compose logs -f --tail=200
  ```
- Réinitialiser les bases locales (destructif) :
  ```bash
  cd infra
  docker compose down -v
  ```

### Sécurité / secrets
- Ne jamais committer de `.env` ni de clés API.
- Masquer les secrets dans les logs et captures d’écran.

## Signaler un problème de sécurité
Si vous pensez avoir trouvé une faille de sécurité, évitez d’ouvrir une issue publique. (Un processus de contact via `SECURITY.md` pourra être ajouté plus tard.)

Merci pour votre contribution !

---

## Développement sous Windows (WSL2 + Docker Desktop)

Cette section décrit une façon pratique de contribuer depuis **Windows** tout en gardant un environnement Linux proche de la prod.

### 1) Installer WSL2 et Ubuntu

1. Ouvrez **PowerShell** (en tant qu’utilisateur normal) et exécutez :
   ```powershell
   wsl --install
   ```
2. Installez **Ubuntu** (si ce n’est pas fait automatiquement) via le Microsoft Store.
3. Vérifiez que votre distro est bien en WSL2 :
   ```powershell
   wsl -l -v
   ```

### 2) Cloner le dépôt dans Ubuntu (WSL)

1. Lancez un terminal **Ubuntu** (Menu Démarrer → “Ubuntu”).
2. Créez un dossier de travail et clonez les dépôts **dans le système de fichiers Linux** (recommandé) :
   ```bash
   mkdir -p ~/dev-canar && cd ~/dev-canar
   git clone https://github.com/Romanovytch/canar.git
   git clone https://github.com/Romanovytch/ragnar.git
   ```

> Recommandation : évitez de travailler dans `/mnt/c/...` (système de fichiers Windows), c’est souvent plus lent et source de soucis de permissions. Préférez `~/dev-canar/...`.

### 3) Ouvrir le projet dans VS Code (Windows) via WSL

1. Installez VS Code sur Windows.
2. Installez l’extension **Remote - WSL**.
3. Depuis le terminal Ubuntu (WSL), ouvrez le projet dans VS Code :
   ```bash
   code .
   ```
VS Code s’ouvre côté Windows, mais exécute le code/terminal dans WSL (indiqué en bas à gauche par “WSL: Ubuntu”).

### 4) Installer Docker Desktop et activer l’intégration WSL

1. Installez **Docker Desktop** sur Windows.
2. Dans Docker Desktop :
   - **Settings** → **Resources** → **WSL Integration**
   - Activez l’intégration pour votre distro (ex. **Ubuntu**)
   - **Apply & Restart**
3. Vérifiez depuis Ubuntu (WSL) :
   ```bash
   docker --version
   docker compose version
   docker run --rm hello-world
   ```

### 5) Lancer Qdrant + Postgres (docker compose) depuis WSL

Depuis le dossier du projet dans WSL :
```bash
cd infra
docker compose up -d
docker compose ps
```

- UI Qdrant : `http://localhost:6333/dashboard`
- Logs :
  ```bash
  docker compose logs -f --tail=200
  ```

Docker Desktop conservera votre docker compose et vous pourrez le lancer directement en appuyant sur le bouton "Play" |>.

**À partir de là, vous pouvez reprendre le guide à [Créer un environnement virtuel](#créer-un-environnement-virtuel)**

**Bon dev, quack !**
