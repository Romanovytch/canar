# Contributing to CanaR

Thanks for taking the time to contribute! This document describes the workflow we use to keep changes reviewable and reproducible.

## Code of Conduct

Be respectful and constructive. (A dedicated `CODE_OF_CONDUCT.md` may be added later.)

## Getting started

### 1) Fork and clone (external contributors)

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/canar.git
   cd canar
   ```
3. Add the upstream remote:
    ```bash
    git remote add upstream https://github.com/Romanovytch/canar.git
    git fetch upstream
    ```

**Maintainers can work directly on the main repository.**

### 2) Local environment

#### Prerequisites
- Python >= 3.10
- Docker + Docker Compose (or Docker Desktop on Windows)
- (Optional) Make

#### Start local dependencies (Qdrant + Postgres)
```bash
cd infra
docker compose up -d
docker compose ps
```

Qdrant UI : https://localhost:6333/dashboard

#### Create a virtual environment
```bash
cd ..
python -m venv .venv
source .venv/bin/activate
```

#### Install CanaR
```bash
pip install -U pip
pip install -e .
```

### 3) Configuration
Create a `.env` file at the project root based on `.env.example`.
> Note: `QDRANT_URL`and `DB_POSTGRES_URL` depend on where CanaR runs.
> - "CanaR on host (venv): use localhost"
> - "CanaR in Docker: use docker service names `qdrant`, `postgres`"

## Workflow

### 1) Create or pick an issue

We track work using Github Issues and project boards:
- CanaR board: https://github.com/users/Romanovytch/projects/2
- AgoRa board: https://github.com/users/Romanovytch/projects/1

When opening an issue, please:
- Assign yourself
- Add labels when possible (at least one `area:*`, one `type:*`, one `priority:*`)
- Describe the problem and expected outcome (you can use provided templates)
- Provide acceptance criteria and dependencies if any

### 2) Create a branch

Create a branch from the current development branch (e.g. `dev`).

Branch naming convention:
- `<type>/<issueNumber>-<short-slug>`
Examples:
- `feat/123-add-default-agent`
- `fix/45-qdrant-timeout`
- `docs/77-update-readme`

Example:
```bash
git switch dev
git pull --rebase
git switch -c feat/123-add-default-agent
```
or shorter (but be careful there is no pull here):
```bash
git switch -c feat/123-add-default-agent dev
```

### 3) Commit messages
Use clear, action-oriented messages. If relevant, reference the issue number:
- `Fix retrieval ranking for multi-collection queries (#123)`
- `Docs: clarify docker hostnames (#77)`

### 4) Run tests locally
Before opening a PR:

```bash
pytest
```
If you use the Makefile:
```bash
make test
```

### 5) Open a Pull Request
Open a PR to the main development branch and fill the PR template.

### 6) Review and merge
- At least **1 approval** (review) is required
- CI checks must pass before merging
- Prefer **Squash and merge** for a clean history (linear history)

## Debugging
- Check docker logs:
    ```bash
    cd infra
    docker compose logs -f --tail=200
    ```
- Reset local databases (destructive):
    ```bash
    cd infra
    docker compose down -v
    ```

## Security / secrets
- Never commit `.env` files or API keys.

