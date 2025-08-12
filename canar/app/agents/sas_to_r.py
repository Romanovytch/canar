from __future__ import annotations

SYSTEM_PROMPT_FR = """Tu es un assistant expert qui traduit du code SAS vers R.
Exigences:
- Produis du R idiomatique, clair et moderne : tidyverse (dplyr, tidyr), readr, purrr.
- Si des fonctions SAS n’ont pas d’équivalent direct, propose une solution R raisonnable.
- Si le code touche à des chemins/fichiers, utilise des chemins génériques (pas d'accès disque réel)
- Réponds en FR. Commence par le code R (bloc ```r), puis une explication sous “Notes”.
"""


def build_messages(user_text: str, sas_code: str | None) -> list[dict]:
    prompt = ""
    if sas_code:
        prompt = f"Voici le code SAS à traduire:\n\n```sas\n{sas_code}\n```\n\n"
        if user_text:
            prompt += f"Contexte/contraintes supplémentaires: {user_text}\n"
    else:
        prompt = f"Demande utilisateur (traduction ou conseil autour de SAS→R):\n{user_text}\n"

    return [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": prompt}
    ]
