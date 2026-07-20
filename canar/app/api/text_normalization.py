from __future__ import annotations

import unicodedata

# APOSTROPHE NORMALIZATION WORKAROUND
# Ollama/bge-m3 returned `unsupported value: NaN` for some French text containing
# typographic apostrophes. Keep this conversion limited to embedding inputs so
# user-visible text is unchanged. To remove the workaround, delete this module
# and the tagged call sites in `embed_client.py` and the benchmark harness.
_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
        "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
        "\u02bc": "'",  # MODIFIER LETTER APOSTROPHE
        "\uff07": "'",  # FULLWIDTH APOSTROPHE
    }
)


def normalize_for_embedding(text: str) -> str:
    """Apply the apostrophe workaround without changing user-visible text."""
    # NFC preserves French accents; the explicit translation is intentionally
    # narrower than NFKC and only converts known apostrophe variants.
    return unicodedata.normalize("NFC", text).translate(_APOSTROPHE_TRANSLATION)
