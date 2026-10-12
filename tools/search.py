"""Search the function index before writing a helper: ``python tools/audit.py --find "..."``.

Ranks the module, class and function rows of ``docs/common/function_index.jsonl`` by the
query's words found in their names (split at ``_`` and capitals), their docstring's first
line and their module's docstring, weighted by how rare a word is (BM25-like). Words match
by a shared stem (``standardise``/``standardization``, ``value``/``values``), and
``SYNONYMS`` maps the abbreviations and the words of the field onto each other: "BH" finds
``benjamini_hochberg``, "MAD" ``RobustStandardize``.

When ``sentence-transformers`` and a local code-embedding model are installed (CodeRankEmbed;
``MORPHOCOVERAGE_EMBEDDING_MODEL`` names it, a path or a model id already downloaded), the
lexical ranking is fused with the embedding similarity. It is optional: without it the
search is lexical and says so, and nothing in CI uses it.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path

import function_index as fi

# Each group: phrases that mean the same thing here. A phrase found in a query or a row adds
# the group's token, so either spelling finds the other.
SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("bh", "benjamini hochberg", "fdr", "false discovery rate", "q value", "multiple testing"),
    ("mad", "median absolute deviation", "robust", "robust scale", "robustly"),
    ("sd", "standard deviation", "std", "stdev"),
    ("se", "sem", "standard error"),
    ("ci", "confidence interval", "interval"),
    ("clopper pearson", "exact binomial", "binomial bound", "binomial interval"),
    ("effect size", "cohen d", "cohen s d", "standardized difference", "hedges g"),
    ("effective sample size", "effective number"),
    ("cv", "coefficient of variation"),
    ("iqr", "interquartile range"),
    ("pca", "principal component", "principal components"),
    ("mmd", "maximum mean discrepancy"),
    ("knn", "nearest neighbour", "nearest neighbor", "neighbours", "neighbors"),
    ("fov", "field of view", "field", "acquisition position"),
    ("object", "cell", "observation"),
    ("repeat", "replicate", "biological repeat"),
    ("p value", "pvalue", "significance"),
    ("percent", "percentage"),
    ("log scale", "logarithm", "log transform"),
    ("bootstrap", "resample", "resampling"),
    ("plural", "pluralize"),
)
WEIGHTS = {"name": 3.0, "owner": 1.0, "summary": 1.0, "module": 0.6, "module summary": 0.3}
_K1, _B = 1.2, 0.5
_QUERY_PROMPT = "Represent this query for searching relevant code: "


def words(text: str) -> list[str]:
    """Lower-case words, ``camelCase`` and ``snake_case`` split, ``'s`` dropped."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text).replace("'s", "")
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def stem(word: str) -> str:
    """A crude stem: British and American spellings joined, plural and verb endings cut."""
    word = re.sub(r"is(e|ed|es|ing|ation|ations)$", r"iz\1", word)
    for ending in ("izations", "ization", "izing", "ized", "izes", "ize"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)] + "iz"
    for ending in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)]
    return word


def _groups() -> list[tuple[str, list[list[str]]]]:
    return [(f"~{group[0].replace(' ', '_')}", [words(p) for p in group]) for group in SYNONYMS]


GROUPS = _groups()


def terms(text: str) -> list[str]:
    """The stems of a text and the synonym groups it mentions."""
    found = words(text)
    out = [stem(w) for w in found]
    for token, phrases in GROUPS:
        for phrase in phrases:
            size = len(phrase)
            if any(found[i : i + size] == phrase for i in range(len(found) - size + 1)):
                out.append(token)
                break
    return out


def _fields(row: dict[str, object], modules: dict[str, str]) -> dict[str, list[str]]:
    name = str(row["name"])
    module = str(row["module"])
    if name == fi.MODULE_ROW:
        return {"name": terms(module.replace(".", " ")), "summary": terms(str(row["summary"]))}
    owner, _, own = name.rpartition(".")
    return {
        "name": terms(own),
        "owner": terms(owner.replace(".", " ")),
        "summary": terms(str(row["summary"])),
        "module": terms(module.replace(".", " ")),
        "module summary": terms(modules.get(module, "")),
    }


def _matches(term: str, vocabulary: set[str]) -> list[str]:
    if term.startswith("~") or len(term) < 5:
        return [term] if term in vocabulary else []
    return [
        w
        for w in vocabulary
        if w == term
        or (len(w) >= 5 and w.startswith(term[:5]) and (w.startswith(term) or term.startswith(w)))
    ]


def lexical(query: str, rows: list[dict[str, object]]) -> list[tuple[float, dict[str, object]]]:
    """The rows matching ``query``, best first, with their score."""
    modules = {str(r["module"]): str(r["summary"]) for r in rows if r["name"] == fi.MODULE_ROW}
    documents = [(_fields(r, modules), r) for r in rows]
    frequency: Counter[str] = Counter()
    for fields, _ in documents:
        frequency.update({t for values in fields.values() for t in values})
    vocabulary = set(frequency)
    count = len(documents)
    lengths = [sum(len(v) for v in f.values()) for f, _ in documents]
    average = sum(lengths) / max(count, 1)
    query_terms = list(dict.fromkeys(terms(query)))
    expanded = {t: _matches(t, vocabulary) for t in query_terms}
    scored = []
    for (fields, row), length in zip(documents, lengths, strict=True):
        score = 0.0
        for matches in expanded.values():
            best = 0.0
            for word in matches:
                tf = sum(WEIGHTS[f] * values.count(word) for f, values in fields.items())
                if not tf:
                    continue
                idf = math.log(1 + (count - frequency[word] + 0.5) / (frequency[word] + 0.5))
                norm = tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * length / average))
                best = max(best, idf * norm)
            score += best
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda s: (-s[0], not s[1]["public"], len(fi.key(s[1]))))
    return scored


def _model_name() -> str | None:
    named = os.environ.get("MORPHOCOVERAGE_EMBEDDING_MODEL")
    if named:
        return named
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--nomic-ai--CodeRankEmbed"
    return "nomic-ai/CodeRankEmbed" if cache.exists() else None


def embedding_ranks(query: str, rows: list[dict[str, object]]) -> list[int] | str:
    """The rank of every row by embedding similarity to ``query``, or why it is unavailable."""
    name = _model_name()
    if name is None:
        return "no local embedding model (set MORPHOCOVERAGE_EMBEDDING_MODEL)"
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return "sentence-transformers is not installed"
    texts = [f"{fi.key(r)} {r.get('signature', '')}\n{r['summary']}" for r in rows]
    digest = hashlib.sha256("\n".join([name, *texts]).encode()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "morphocoverage" / f"index-{digest}.npy"
    model = SentenceTransformer(name, trust_remote_code=True, local_files_only=True)
    if cache.exists():
        vectors = np.load(cache)
    else:
        vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    wanted = model.encode([_QUERY_PROMPT + query], normalize_embeddings=True)[0]
    order = np.argsort(-(vectors @ wanted), kind="stable")
    ranks = [0] * len(rows)
    for rank, position in enumerate(order, start=1):
        ranks[int(position)] = rank
    return ranks


def find(
    query: str, rows: list[dict[str, object]] | None = None, embeddings: bool = True
) -> tuple[list[dict[str, object]], str]:
    """The rows for ``query``, best first, and a note on how they were ranked."""
    rows = fi.read_index() if rows is None else rows
    lexical_order = [row for _, row in lexical(query, rows)]
    ranks = embedding_ranks(query, rows) if embeddings else "embeddings not asked for"
    if isinstance(ranks, str):
        return lexical_order, f"lexical ranking ({ranks})"
    position = {id(r): i for i, r in enumerate(lexical_order, start=1)}
    # Reciprocal-rank fusion: a row both rankings place high wins over one only one does.
    fused = sorted(
        range(len(rows)),
        key=lambda i: (
            -(
                1 / (60 + ranks[i])
                + (1 / (60 + position[id(rows[i])]) if id(rows[i]) in position else 0)
            )
        ),
    )
    return [rows[i] for i in fused], "lexical and embedding ranking fused"


def show(query: str, limit: int = 15) -> str:
    """The ``--find`` output: the best rows, one per line."""
    found, note = find(query)
    lines = [f"{note}; {len(found)} matches for {query!r}"]
    for row in found[:limit]:
        lines.append(f"  {fi.key(row)}  [{row['kind']}]  {row['summary']}")
    return "\n".join(lines)
