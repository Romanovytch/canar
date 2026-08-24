"""The markdown-table workaround: what it rewrites, and what it must leave alone.

The embedding server answers 500 for some markdown tables and loses the answer
relevancy score for that question. The failure is deterministic, so RAGAS's
retries cannot recover it — the text has to change before it is sent.
"""

from __future__ import annotations

from embedding_normalization import flatten_markdown_tables, prepare_for_embedding

TABLE = "| Format | Seuil |\n|---|---|\n| CSV | 5 Go |"


def test_a_table_becomes_plain_text():
    assert flatten_markdown_tables(TABLE) == "Format Seuil\nCSV 5 Go"


def test_the_separator_row_is_dropped():
    assert "---" not in flatten_markdown_tables(TABLE)


def test_alignment_markers_are_dropped_too():
    table = "| a | b |\n|:---|---:|\n| c | d |"
    assert flatten_markdown_tables(table) == "a b\nc d"


def test_cell_contents_survive():
    """Flattening is for the server, not for the reader: no word may be lost."""
    flat = flatten_markdown_tables(TABLE)
    for word in ("Format", "Seuil", "CSV", "5", "Go"):
        assert word in flat


def test_prose_is_untouched():
    prose = "utilitR recommends open_dataset rather than read_parquet.\nIt matters."
    assert flatten_markdown_tables(prose) == prose


def test_a_lone_pipe_is_not_a_table():
    line = "run `a | b` in the shell"
    assert flatten_markdown_tables(line) == line


def test_text_around_a_table_is_kept():
    text = "Before.\n" + TABLE + "\nAfter."
    flat = flatten_markdown_tables(text)
    assert flat.startswith("Before.")
    assert flat.endswith("After.")


def test_the_apostrophe_workaround_still_applies():
    """Both workarounds run, and the older one is not lost to the newer."""
    assert "’" not in prepare_for_embedding("l’utilisateur")
