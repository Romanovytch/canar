# Setup — running the 3-strategy benchmark (dense, sparse, hybrid)

Assumes you already work on the project (CanaR + AgoRa cloned, venvs, Qdrant and
Ollama running, the dense benchmark working). This is just the **delta** to also
run `simple_sparse` and `hybrid`: the collection needs **named dense + sparse
vectors**, which is a one-time env + ingestion setup. It lives outside git
(local venvs, `.env`, the Qdrant collection), so each machine does it once.

Commands assume you are in `canar/benchmark/`; `../../agora` is the AgoRa repo.

## 1. Update AgoRa (brings multi-vector ingestion)

The dense+sparse ingestion landed on AgoRa's `dev` (PR #2). Pull it:

```bash
cd ../../agora            # the agora repo next to canar
git checkout dev && git pull
```

## 2. Install fastembed (sparse/BM25 encoder)

Needed by AgoRa (to build sparse vectors) and by CanaR/benchmark (to embed
sparse queries):

```bash
../../agora/.venv/bin/pip install "fastembed>=0.8.0"
.venv/bin/pip install "fastembed>=0.8.0"     # the benchmark venv
```

## 3. Declare named vectors in `agora/sources.yaml`

Add a `vector_index` block so AgoRa builds both vectors per chunk:

```yaml
vector_index:
  vectors:
    - name: dense
      kind: dense
    - name: sparse
      kind: sparse
      model: Qdrant/bm25
```

## 4. Ingest a multi-vector collection

For results comparable across the team, pin the utilitR source to the same
commit (the one this collection was built from), then build a **new** collection
(`utilitr_v2`) so the existing dense-only one stays as a fallback:

```bash
# pin the docs source (path is your utilitR checkout — see sources.yaml repo_path)
git -C ../../utilitR checkout 490a56d

cd ../../agora && source .venv/bin/activate
agora-ingest \
  --sources-config-path sources.yaml \
  --source utilitr \
  --collection utilitr_v2 \
  --dotenv-path .env \
  --drop-collection
```

Every chunk records the source commit in its payload (`git_commit`), so a
collection is auditable: a teammate can confirm theirs was built from `490a56d`.

Verify it has both vectors:

```bash
curl -s http://localhost:6350/collections/utilitr_v2 \
  | python3 -c "import sys,json; p=json.load(sys.stdin)['result']['config']['params']; print('dense:', p.get('vectors')); print('sparse:', list((p.get('sparse_vectors') or {}).keys()))"
```

## 5. Point `canar/.env` at the new collection

```env
QDRANT_COLLECTIONS=utilitr_v2
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
FASTEMBED_SPARSE_MODEL=Qdrant/bm25
```

Keep `environment.collection` in `benchmark/config.yaml` matching
(`utilitr_v2`) — the preflight aborts on a mismatch.

## 6. Run the benchmark

```bash
cd ../canar/benchmark && source .venv/bin/activate
python e2e/eval_e2e.py
```

It runs the dataset once per strategy (`dense`, `sparse`, `hybrid`) and prints a
comparison table. Each run is saved as one folder:

```
e2e/results/run_<timestamp>/
├── comparison.csv      the three strategies side by side
├── dense/   (metrics.csv + answers.md)
├── sparse/  (metrics.csv + answers.md)
└── hybrid/  (metrics.csv + answers.md)
```

For a quick pass, set `limit: 3` in `config.yaml`. See `REPRODUCIBILITY.md` for
what reproduces exactly and what is only indicative.

## Fallback (dense-only)

If sparse/hybrid misbehave, revert to the dense collection in `canar/.env`:

```env
QDRANT_COLLECTIONS=utilitr_v1
# comment out QDRANT_DENSE_VECTOR_NAME, QDRANT_SPARSE_VECTOR_NAME, FASTEMBED_SPARSE_MODEL
```

Set `environment.collection: utilitr_v1` in `config.yaml` and keep only the
`dense` profile.
