[🇫🇷 Français](CONTRIBUTING.fr.md) | [🇬🇧 English](CONTRIBUTING.md)

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

or `make up` if you use Makefile.

Qdrant UI : https://localhost:6333/dashboard

#### Create a virtual environment
```bash
cd ..
python -m venv .venv
source .venv/bin/activate
```

or `make venv` if you use Makefile (you only need to do this once to create the venv).

#### Install CanaR
```bash
pip install -U pip
pip install -e ".[dev]"
```

> `pip install -e ".[dev]"` will install dev tools such as pytest and ruff for testing and formatting.

or `make install` if you use Makefile

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
- Use one of the templates: Epic, Subtask, Feature request, Documentation, Bug report
> Epics are for big features or multiple features that needs to be broke down into subtasks. Subtasks should be created as new issues and then linked to its Epic.
- Assign yourself if you take it
- Add labels when possible (at least one `area:*`, one `type:*`, one `priority:*`)
- Fill the templates' inputs.


### 2) Create a branch

Create a branch from the current development branch (e.g. `dev`). If it's an Epic's subtask, create the branch from the epic branch.

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

### 4) Run tests, lint and format check localy

#### Tests

Before opening a PR:
```bash
pytest
```
or `make test`

#### Lint & Format check

Ruff can check code quality for errors (also called *lint*):
```bash
ruff check .
```
or `make lint`

Check if files are well formated:
```bash
ruff format --check
```
or `make format-check`

> Auto format :
> ```bash
> ruff check . --fix
> ```
> or `make format`

If you use makefile, `make ci` does it all just like github CI.

### 5) Open a Pull Request
Open a PR to the dev branch (or epic branch if subtask) and fill the PR template.

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

