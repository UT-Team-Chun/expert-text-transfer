#!/usr/bin/env python
"""PIVOTAL GATE (peer-review response): is the text a *geological prior* or a
*paraphrase of the SPT-N label*?

A reviewer's biggest reject risk: borehole descriptions contain consistency/density
words ("loose/dense/stiff/very stiff", 緩い/硬い/N値) that are near-direct proxies for
SPT-N, so a frozen-LM text feature might just be reading the answer off the text. We
isolate genuine *lithological* content with three controls, each plugged into the
same leak-proof harness (scripts.uk_transfer_test._evaluate_lro: per-fold PCA on train
regions, multi-seed, shuffled-embedding null, sign test):

  full          : LM embedding of the unmodified description
  lithology_only: LM embedding after stripping strength/density/hardness/N/penetration/
                  refusal/numeric tokens  -> keeps grain size, colour, lithology nouns
  hardness_only : LM embedding after stripping lithology nouns -> keeps consistency words
  dictionary    : rule-based lithology/regime one-hot (no LM at all)
  tfidf         : TF-IDF char n-grams (no LM, no semantics)

If the content effect SURVIVES on lithology_only (and on dictionary/tfidf it is weaker
than LM), the transferable signal is genuine geological narrative, not an N-value
read-off. If dictionary ~= LM, the honest reframe is "expert text carries transferable
information; the frozen embedding is a convenient high-coverage representation."

CPU only. Usage:
  python -m scripts.text_leakage_controls --domain japan --out <json> --cache-dir <dir>
  python -m scripts.text_leakage_controls --domain uk --out <json> --cache-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.uk_transfer_test import _evaluate_lro, embed_texts

LOG = logging.getLogger("text_leakage_controls")
REPO = Path(__file__).resolve().parents[2]

# ---- UK BS5930 descriptor vocab -------------------------------------------------
# Consistency/relative-density terms = the SPT-N proxy to strip for lithology_only.
# v2 (2026-07-04): +consolidation state, friable/indurated, manual-test phrases,
# after the independent residual-cue audit (2026-07-04_leakage_audit.json).
_UK_STRENGTH = (r"\b(?:very\s+loose|loose|medium\s+dense|very\s+dense|dense|"
                r"very\s+soft|soft|firm\s+to\s+stiff|firm|stiff\s+to\s+very\s+stiff|"
                r"very\s+stiff|stiff|very\s+hard|hard|extremely\s+weak|very\s+weak|"
                r"moderately\s+weak|weak|moderately\s+strong|very\s+strong|strong|"
                r"poorly\s+cemented|well\s+cemented|cemented|refusal|compact|"
                r"(?:over|un|normally\s+)?consolidat\w*|friable|indurat\w*|"
                r"crumbl\w*|breaks?\s+easily|finger\s+pressure|hammer(?:\s+blow)?)\b")
_UK_NUM = r"\bN\s*=\s*-?\d+\b|\d+(?:\.\d+)?\s*(?:mm|cm|m)\b|\d+(?:\.\d+)?"
# Lithology nouns (BS5930 capitalises the principal soil/rock) to strip for hardness_only.
_UK_LITHO = (r"\b(?:CLAY|SAND|GRAVEL|SILT|SANDSTONE|MUDSTONE|SILTSTONE|CLAYSTONE|"
             r"LIMESTONE|DOLOMITE|CHALK|PEAT|MARL|TILL|TOPSOIL|MADE\s+GROUND|COBBLES|"
             r"BOULDERS|clay|sand|gravel|silt)\b")
_uk_strength_re = re.compile("(?:%s)|(?:%s)" % (_UK_STRENGTH, _UK_NUM), re.IGNORECASE)
_uk_litho_re = re.compile(_UK_LITHO)

# ---- Japanese descriptor vocab --------------------------------------------------
# Strength/consistency/penetration/N tokens to strip for lithology_only.
# v2 (2026-07-04): extended after an independent broader-vocabulary audit
# (2026-07-04_leakage_audit.json) found residual mechanical-state cues in ~26% of
# v1-stripped texts: NOMINAL forms (締まり/硬さ), the degree word 中位
# (context-checked: 「含水中位」「中位の締まり具合」), cementation state
# (固結/未固結/溶結), and manual strength-test phrases (指圧で崩せる/ハンマーの
# 軽打/爪跡がつく/手で崩せる/ナイフで削れる). 粘性(?!土) preserves the lithology
# noun 粘性土. v1 numbers remain in the provenance JSONs; the manuscript quotes v2.
STRIP_VOCAB_VERSION = "v2"
_JP_STRENGTH = (r"非常に緩|やや緩|緩い|緩く|緩んで|ゆるい|ゆるく|密な|密に|締まった|締まっ|"
                r"よく締|硬い|硬く|硬質|軟質|軟らか|軟弱|N値|Ｎ値|標準貫入|貫入|打撃回数|打撃|"
                r"\bN\s*=?\s*\d+|含水(?:大|多|高)|高位|"
                r"締ま?り|中位|固結|溶結|密実|堅硬|硬岩|軟岩|硬さ|軟さ|コンシステンシー?|"
                r"指圧|指で押|指で潰|手で崩|爪跡|ハンマー|軽打|ナイフで削|粘着性|粘性(?!土)")
_JP_NUM = r"\d+(?:\.\d+)?"
# Lithology nouns to strip for hardness_only.
_JP_LITHO = (r"火山灰|軽石|アスファルト|シルト|ローム|粘性土|砂質|礫質|粗砂|中砂|細砂|砂岩|"
             r"泥岩|凝灰岩|花崗岩|安山岩|玄武岩|石灰岩|砂|礫|粘土|泥|岩|コア|貝殻|有機質|腐植|軽石")
_jp_strength_re = re.compile("(?:%s)|(?:%s)" % (_JP_STRENGTH, _JP_NUM))
_jp_litho_re = re.compile(_JP_LITHO)


def strip_text(text: str, domain: str, mode: str, *,
               header_terms: frozenset[str] | None = None,
               template_terms: Sequence[str] | None = None) -> str:
    """Strip a single description according to ``mode``.

    ``full`` / ``lithology_only`` / ``hardness_only`` are FROZEN: their output
    is byte-identical to the pre-P-T5 implementation, because
    ``STRIP_VOCAB_VERSION`` and the stripped text itself participate in the
    embedding cache keys of every completed analysis.

    ``lithology_only_depersonalised`` (P-T5) additionally removes identity
    tokens -- see :func:`depersonalise_text`. ``header_terms`` /
    ``template_terms`` supply the per-borehole and per-project context those
    operations need; without them only the context-free operations
    (normalisation + rule-based place names) run. The batch entry point that
    assembles the context is :func:`apply_strip_mode`.
    """
    if mode == "full":
        return text
    if mode == DEPERSONALISE_MODE:
        base = strip_text(text, domain, "lithology_only")
        return depersonalise_text(base, header_terms=header_terms,
                                  template_terms=template_terms)
    if domain == "uk":
        rx = _uk_strength_re if mode == "lithology_only" else _uk_litho_re
    else:
        rx = _jp_strength_re if mode == "lithology_only" else _jp_litho_re
    return rx.sub(" ", text)


# =================================================================================
# P-T5 -- proper-noun strip + template normalisation
# ---------------------------------------------------------------------------------
# Pre-registered control (docs/research/2026-08-11_nc_text_preregistration.md):
# "固有名詞 strip（調査名・地名 token 除去）+ テンプレート正規化後も効果残存 |
#  減衰 <=5 pt かつ負 in >=7/8".
#
# The reviewer objection this answers: the embedding might be reading WHO wrote
# the log or WHERE it was written (project name, town name, contractor house
# phrasing) rather than the geology of the layer. Two complementary operations:
#
#   (a) proper-noun strip
#       a1 data-driven, per borehole: delete any substring of length >= 2 that
#          occurs in that borehole's OWN archive header (survey/project name,
#          commissioning agency, survey contractor). No hand-built name list.
#       a2 rule-based place names: delete a maximal run of kanji/katakana that
#          immediately precedes an administrative/geographic suffix, together
#          with the suffix.
#   (b) template normalisation: delete substrings shared by >= 90% of a
#       project's layer texts (per-project boilerplate carries no
#       layer-specific information but does identify the project), after
#       collapsing full-width/half-width and whitespace variants so trivially
#       different templates collapse together.
#
# Both are exposed independently (``ops=``) and each records how much text it
# removed, because a strip that empties layers is a deletion, not a control.
# =================================================================================

DEPERSONALISE_MODE = "lithology_only_depersonalised"
# Separate from STRIP_VOCAB_VERSION ON PURPOSE: bumping the shared constant
# would invalidate every cached embedding of the frozen modes.
DEPERSONALISE_VERSION = "d1"
STRIP_MODES = ("full", "lithology_only", "hardness_only", DEPERSONALISE_MODE)

#: Archive header fields searched for identity substrings (a1).
HEADER_FIELDS = ("survey_name", "project_name", "orderer_name", "surveyor_name")
HEADER_MIN_NGRAM = 2      # "any substring of length >= 2" (prereg wording)
HEADER_MAX_NGRAM = 16     # longest matched chunk; longer names fall out piecewise
#: Template normalisation (b).
TEMPLATE_MIN_DOC_FRACTION = 0.9   # "high fraction" of a project's layers
TEMPLATE_MIN_LEN = 6              # shorter shared strings are ordinary vocabulary
TEMPLATE_MAX_LEN = 120
TEMPLATE_MIN_PROJECT_LAYERS = 5   # below this, "90% of layers" is meaningless
TEMPLATE_MAX_CANDIDATES = 20_000  # safety valve on the level-wise search
TEMPLATE_MAX_TEXTS_SCANNED = 4_000
#: Rule-based place names (a2).
PLACE_MIN_STEM = 2   # 1-char stems are where the geology lives (火|山, 河|川)
PLACE_MAX_STEM = 8

DEPERSONALISE_OPS = ("normalise", "header", "place", "template")
METADATA_PARQUET = REPO / "data/features/derived/kunijiban_metadata.parquet"

# --- whitespace / width normalisation --------------------------------------------
# KuniJiban layer narratives carry LITERAL backslash-n line breaks (55% of rows
# in soil_text_layers.csv), not real newlines, plus full-width percent/space.
_ESCAPED_WS_RE = re.compile(r"\\+[nrt]")
_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """NFKC + whitespace collapse. Idempotent."""
    t = unicodedata.normalize("NFKC", str(text))
    t = _ESCAPED_WS_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


# --- geological vocabulary that the strip must NEVER damage -----------------------
# Chosen from the real corpus, not from imagination: a 300k-layer scan of
# data/features/derived/soil_text_layers.csv shows the suffix characters are
# dominated by GEOLOGY, not by place names -- 山 occurs 30,624 times of which
# 火山* (14,305) and 安山* (10,072) alone are 80%; 川 is 51% 河川/河道; 区 is 90%
# 区分 (岩級区分/硬軟区分). These spans are masked out (replaced by private-use
# placeholders) before either proper-noun operation runs and restored after, so
# they can neither be matched by a header n-gram nor absorbed into a place-name
# stem. Cost: a place name that ends inside a protected term survives (e.g. the
# river 小砂川 keeps its 砂). Deliberate -- a strip that ate 火山灰 would destroy
# the signal the paper is about and would look like a clean null.
_GEO_PROTECT_TERMS: tuple[str, ...] = (
    # 山 (火山* 17,629 + 安山* 12,471 + 地山 217 + 山砂* 378 dominate; the
    #     landform words 山麓/山地/山側/鉱山 occur <=2 times each and carry no
    #     lithology, so they are left unprotected to keep place-name recall)
    "火山灰質", "火山噴出物", "火山灰", "火山礫", "火山岩", "火山砂", "火山", "安山岩",
    "安山", "地山", "山砂利", "山砂", "山土",
    # 川 (河川 219, 河道 29, 川砂 35)
    "河川堆積物", "河川敷", "河川", "旧河道", "現河道", "河道", "川砂利", "川砂",
    # 島 湖 橋 (半島 0 / 諸島 5 are place morphology -> NOT protected)
    "島状", "湖成", "湖沼", "湖底", "桟橋", "橋台", "橋脚",
    # 郡 (層郡 13 = a 層群 typo, 三郡 5 = 三郡変成岩)
    "層郡", "三郡",
    # geological time (without these, 第三紀島尻層群 loses 第三紀 to the 島 rule)
    "新第三紀", "古第三紀", "第三紀", "第四紀", "更新世", "完新世", "沖積世", "洪積世",
    "白亜紀", "ジュラ紀", "三畳紀", "中生代", "古生代", "新生代",
    # 工区 / 地区 (the 区分 family dominates 区 in this corpus)
    "区分", "区間",
    # lithology / fabric vocabulary the header rule must not eat
    "砂利", "砂質", "砂岩", "砂礫", "砂混じり", "細砂", "中砂", "粗砂", "微砂",
    "礫質", "礫岩", "礫混じり", "細礫", "中礫", "粗礫", "円礫", "角礫", "亜角", "亜円",
    "玉石", "粘性土", "粘土", "シルト", "ローム", "軽石", "凝灰", "泥岩", "泥炭",
    "頁岩", "花崗", "玄武", "石灰", "チャート", "シラス", "腐植", "有機質", "貝殻",
    "風化", "含水", "淘汰", "段丘", "沖積", "洪積", "堆積", "埋土", "盛土", "表土",
    "光沢", "条線", "亀裂", "破砕", "節理", "層理", "互層", "分級", "舗装",
)
_geo_protect_re = re.compile(
    "|".join(re.escape(t) for t in
             sorted(set(_GEO_PROTECT_TERMS), key=lambda s: (-len(s), s))))
_PUA_BASE = 0xE000
_PUA_SLOTS = 0x1000


def _mask_protected(text: str) -> tuple[str, list[str]]:
    """Replace protected geological spans by private-use placeholders."""
    out: list[str] = []
    saved: list[str] = []
    pos = 0
    for m in _geo_protect_re.finditer(text):
        if len(saved) >= _PUA_SLOTS:  # pathological input; stop masking
            break
        out.append(text[pos:m.start()])
        out.append(chr(_PUA_BASE + len(saved)))
        saved.append(m.group(0))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), saved


def _unmask(text: str, saved: list[str]) -> str:
    if not saved:
        return text
    hi = _PUA_BASE + len(saved)
    return "".join(saved[ord(c) - _PUA_BASE] if _PUA_BASE <= ord(c) < hi else c
                   for c in text)


# --- (a2) rule-based place names --------------------------------------------------
# Supported suffixes. Every one of these was checked against the corpus scan
# above; the REJECTED set below is the part of the prereg's "and similar" that
# would have cost geology.
_PLACE_SUFFIX_SINGLE = "都府県市町村郡川山島湖港駅橋"
_PLACE_SUFFIX_MULTI = ("工区", "地区", "団地", "地内", "地先")
#: suffix -> why it is NOT applied (each is pinned by a test).
_PLACE_SUFFIX_REJECTED = {
    "道": "河道/旧河道 are geomorphic and 舗装道路・坑道・農道 are descriptive; "
          "the corpus has 23 旧河道 to 9 国道",
    "線": "条線 (slickenside striae) is 68% of all 線 occurrences; "
          "境界線/直線/弱線 are descriptive",
    "区": "区分 (岩級区分/硬軟区分/酸化区分) is 90% of all 区 occurrences",
    "沢": "光沢 (lustre: 金属光沢/樹脂光沢/絹糸光沢) is 84% of all 沢 occurrences",
    "丘": "段丘 (river terrace) is 92% of all 丘 occurrences; 砂丘 is a landform",
    "谷": "埋積谷/谷底低地 are landform descriptions",
    "台": "橋台/土台 are structures, not place names",
}
# Name characters: kanji, 々, katakana (incl. the long vowel mark, excl. ・).
_NAME_CH = r"[一-鿿々ァ-ヺー]"
_place_re = re.compile(
    rf"{_NAME_CH}{{{PLACE_MIN_STEM},{PLACE_MAX_STEM}}}"
    rf"(?:[{_PLACE_SUFFIX_SINGLE}]|(?:{'|'.join(_PLACE_SUFFIX_MULTI)}))(?!分|間)"
)


def strip_place_names(text: str, *, protect_geology: bool = True) -> str:
    """(a2) Delete ``<kanji/katakana run><admin-or-geographic suffix>``.

    The run must be at least ``PLACE_MIN_STEM`` characters: the geological
    traps are almost all 1-character stems (火|山, 安|山, 河|川), so the
    minimum plus the protected-vocabulary mask makes the rule safe. The price
    is that 1-character place stems (荒川, 谷川) survive -- they are caught by
    the data-driven header strip whenever they appear in the survey name,
    which is where they realistically come from.
    """
    masked, saved = _mask_protected(text) if protect_geology else (text, [])
    return _unmask(_place_re.sub(" ", masked), saved)


# --- (a1) data-driven, per-borehole header strip ----------------------------------


def header_terms_for_fields(fields: Iterable[object], *,
                            min_n: int = HEADER_MIN_NGRAM,
                            max_n: int = HEADER_MAX_NGRAM) -> frozenset[str]:
    """All length-``[min_n, max_n]`` substrings of one borehole's header fields."""
    grams: set[str] = set()
    for f in fields:
        if f is None or (isinstance(f, float) and math.isnan(f)):
            continue
        s = normalise_text(str(f))
        if not s:
            continue
        n = len(s)
        for k in range(min_n, min(max_n, n) + 1):
            for i in range(n - k + 1):
                g = s[i:i + k]
                if " " in g:
                    continue
                if not any(ch.isalnum() for ch in g):
                    continue  # pure punctuation
                grams.add(g)
    return frozenset(grams)


def _script_class(ch: str) -> str:
    """Cohesive script classes: a match may not start or end INSIDE one.

    Kanji and punctuation are 'splittable' -- Japanese writes compounds
    without spaces, so a header n-gram legitimately covers part of a kanji
    run. Katakana / hiragana / latin runs are single lexical words, and a
    2-character header n-gram that lands inside one is a coincidence, not an
    identity token: on the real corpus this is what turned マトリックス into
    "マトリ ス" (matrix), コンクリート into " クリート" (concrete) and
    〜からなる into " なる".
    """
    o = ord(ch)
    if 0x30A1 <= o <= 0x30FA or ch in "ーヽヾ゛゜":
        return "katakana"
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    if ch.isascii() and ch.isalnum():
        return "latin"
    return ""          # kanji, symbols, spaces -> always splittable


def _splits_a_word(text: str, i: int, j: int) -> bool:
    left = _script_class(text[i])
    if i > 0 and left and _script_class(text[i - 1]) == left:
        return True
    right = _script_class(text[j - 1])
    return j < len(text) and bool(right) and _script_class(text[j]) == right


def strip_header_terms(text: str, terms: frozenset[str] | None, *,
                       max_n: int = HEADER_MAX_NGRAM,
                       min_n: int = HEADER_MIN_NGRAM,
                       protect_geology: bool = True,
                       word_boundary: bool = True) -> str:
    """(a1) Greedy longest-match deletion of this borehole's header substrings.

    Guarded twice: protected geological spans are masked out first, and a
    candidate match that would split a katakana/hiragana/latin word is
    rejected (see :func:`_splits_a_word`).
    """
    if not terms:
        return text
    masked, saved = _mask_protected(text) if protect_geology else (text, [])
    out: list[str] = []
    i, n = 0, len(masked)
    while i < n:
        hit = 0
        for k in range(min(max_n, n - i), min_n - 1, -1):
            if masked[i:i + k] in terms:
                if word_boundary and _splits_a_word(masked, i, i + k):
                    continue
                hit = k
                break
        if hit:
            out.append(" ")
            i += hit
        else:
            out.append(masked[i])
            i += 1
    return _unmask("".join(out), saved)


def header_terms_by_borehole(
    boring_files: Iterable[str],
    *,
    metadata: pd.DataFrame | None = None,
    metadata_path: Path = METADATA_PARQUET,
    fields: Sequence[str] = HEADER_FIELDS,
    max_n: int = HEADER_MAX_NGRAM,
) -> dict[str, frozenset[str]]:
    """``boring_file -> header n-gram set`` from the archive header table."""
    want = {str(b) for b in boring_files}
    if metadata is None:
        cols = ["boring_file", *fields]
        metadata = pd.read_parquet(metadata_path, columns=cols)
    meta = metadata.copy()
    meta["boring_file"] = meta["boring_file"].astype(str)
    meta = meta[meta["boring_file"].isin(want)]
    use = [c for c in fields if c in meta.columns]
    out: dict[str, frozenset[str]] = {}
    for row in meta[["boring_file", *use]].itertuples(index=False):
        terms = header_terms_for_fields(row[1:], max_n=max_n)
        if terms:
            out[row[0]] = terms
    return out


# --- (b) template normalisation ----------------------------------------------------


def frequent_substrings(texts: Sequence[str], *,
                        min_fraction: float = TEMPLATE_MIN_DOC_FRACTION,
                        min_len: int = TEMPLATE_MIN_LEN,
                        max_len: int = TEMPLATE_MAX_LEN,
                        max_candidates: int = TEMPLATE_MAX_CANDIDATES,
                        ) -> list[str]:
    """Maximal substrings present in >= ``min_fraction`` of ``texts``.

    Level-wise (Apriori over character n-grams): a length-(k+1) substring can
    only be frequent if its length-k prefix is, so each level is generated from
    the survivors of the previous one. Returns only MAXIMAL survivors (a
    frequent substring that is contained in a longer frequent one is dropped),
    longest first -- the order the deletion pass needs.
    """
    texts = [t for t in texts]
    n = len(texts)
    if n == 0:
        return []
    need = max(1, int(math.ceil(min_fraction * n - 1e-9)))
    k = min_len
    df: Counter[str] = Counter()
    for t in texts:
        df.update({t[i:i + k] for i in range(len(t) - k + 1)})
    cur = {s for s, c in df.items() if c >= need}
    survivors: list[str] = []
    while cur and k < max_len:
        if len(cur) > max_candidates:  # pathological project; keep what we have
            break
        survivors.extend(cur)
        nxt: Counter[str] = Counter()
        for t in texts:
            grown = {t[i:i + k + 1] for i in range(len(t) - k)
                     if t[i:i + k] in cur}
            nxt.update(grown)
        k += 1
        cur = {s for s, c in nxt.items() if c >= need}
    survivors.extend(cur)
    if not survivors:
        return []
    ordered = sorted(set(survivors), key=lambda s: (-len(s), s))
    maximal: list[str] = []
    for s in ordered:
        if not any(s in longer for longer in maximal):
            maximal.append(s)
    return maximal


def template_terms_by_project(
    texts: Sequence[str],
    project_keys: Sequence[object],
    *,
    min_fraction: float = TEMPLATE_MIN_DOC_FRACTION,
    min_len: int = TEMPLATE_MIN_LEN,
    min_layers: int = TEMPLATE_MIN_PROJECT_LAYERS,
    max_texts: int = TEMPLATE_MAX_TEXTS_SCANNED,
    dedupe_keys: Sequence[object] | None = None,
) -> dict[object, list[str]]:
    """``project_key -> boilerplate substrings`` to delete.

    Document frequency is computed over DISTINCT layer texts (optionally
    per ``dedupe_keys``, e.g. ``boring_file``): the analysis frame is one row
    per SPT measurement, so the same layer narrative repeats across rows and
    would otherwise inflate the frequency of whatever the deepest boreholes
    happen to say.
    """
    by_project: dict[object, set] = {}
    for i, (t, p) in enumerate(zip(texts, project_keys)):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        key = t if dedupe_keys is None else (dedupe_keys[i], t)
        by_project.setdefault(p, set()).add(key)
    out: dict[object, list[str]] = {}
    for p, keys in by_project.items():
        uniq = sorted(k if dedupe_keys is None else k[1] for k in keys)
        if len(uniq) < min_layers:
            continue
        if len(uniq) > max_texts:  # deterministic thinning, detection only
            step = len(uniq) / max_texts
            uniq = [uniq[int(i * step)] for i in range(max_texts)]
        terms = frequent_substrings(uniq, min_fraction=min_fraction,
                                    min_len=min_len)
        if terms:
            out[p] = terms
    return out


def remove_template_terms(text: str, terms: Sequence[str] | None) -> str:
    """(b) Delete this project's boilerplate, longest term first."""
    if not terms:
        return text
    for term in sorted(terms, key=len, reverse=True):
        if term:
            text = text.replace(term, " ")
    return text


# --- the composed operation --------------------------------------------------------


def depersonalise_text(text: str, *,
                       header_terms: frozenset[str] | None = None,
                       template_terms: Sequence[str] | None = None,
                       ops: Sequence[str] = DEPERSONALISE_OPS,
                       protect_geology: bool = True,
                       header_max_n: int = HEADER_MAX_NGRAM) -> str:
    """Apply the P-T5 operations in ``ops`` order. Idempotent.

    Every operation substitutes a SPACE (never the empty string) and ends with
    a whitespace collapse, so a deletion can never fuse its two neighbours into
    a token that a later pass -- or a second application -- would match.
    """
    t = text
    for op in ops:
        if op == "normalise":
            t = normalise_text(t)
        elif op == "header":
            t = strip_header_terms(t, header_terms, max_n=header_max_n,
                                   protect_geology=protect_geology)
        elif op == "place":
            t = strip_place_names(t, protect_geology=protect_geology)
        elif op == "template":
            t = remove_template_terms(t, template_terms)
        else:
            raise ValueError(f"unknown depersonalise op {op!r}; "
                             f"expected a subset of {DEPERSONALISE_OPS}")
        t = normalise_text(t)
    return t


def _removal_stats(before: Sequence[str], after: Sequence[str]) -> dict:
    chars_before = sum(len(t) for t in before)
    chars_after = sum(len(t) for t in after)
    removed = chars_before - chars_after
    return {
        "frac_chars_removed": round(removed / max(chars_before, 1), 4),
        "chars_removed": int(removed),
        "n_texts_changed": int(sum(a != b for a, b in zip(before, after))),
        "n_emptied": int(sum(len(a) == 0 and len(b) > 0
                             for a, b in zip(after, before))),
    }


def depersonalise_corpus(
    texts: Sequence[str],
    *,
    boring_files: Sequence[str] | None = None,
    project_keys: Sequence[object] | None = None,
    metadata: pd.DataFrame | None = None,
    metadata_path: Path = METADATA_PARQUET,
    ops: Sequence[str] = DEPERSONALISE_OPS,
    fields: Sequence[str] = HEADER_FIELDS,
    template_min_fraction: float = TEMPLATE_MIN_DOC_FRACTION,
    protect_geology: bool = True,
) -> tuple[list[str], dict]:
    """Run the P-T5 operations over a corpus and report what each removed.

    ``boring_files`` unlocks the data-driven header strip and (via the archive
    header table) the ``project_key`` used for template normalisation;
    ``project_keys`` overrides the latter. Operations run in ``ops`` order and
    each is measured separately, so the caller can see -- BEFORE committing
    compute -- how many layers a given operation empties.
    """
    ops = tuple(ops)
    unknown = set(ops) - set(DEPERSONALISE_OPS)
    if unknown:
        raise ValueError(f"unknown depersonalise op(s) {sorted(unknown)}; "
                         f"expected a subset of {DEPERSONALISE_OPS}")
    cur = [str(t) for t in texts]
    stats: dict = {"mode": DEPERSONALISE_MODE, "version": DEPERSONALISE_VERSION,
                   "ops": list(ops), "n_texts": len(cur),
                   "chars_in": sum(len(t) for t in cur), "per_op": {}}

    header_map: dict[str, frozenset[str]] = {}
    if "header" in ops and boring_files is not None:
        header_map = header_terms_by_borehole(
            boring_files, metadata=metadata, metadata_path=metadata_path,
            fields=fields)
        stats["header_coverage"] = round(
            float(np.mean([str(b) in header_map for b in boring_files])), 4)
        stats["n_boreholes_with_header"] = len(header_map)

    if "place" in ops:
        stats["place_suffixes"] = {
            "single": list(_PLACE_SUFFIX_SINGLE),
            "multi": list(_PLACE_SUFFIX_MULTI),
            "rejected": dict(_PLACE_SUFFIX_REJECTED),
            "min_stem": PLACE_MIN_STEM,
        }

    for op in ops:
        before = cur
        if op == "normalise":
            cur = [normalise_text(t) for t in before]
        elif op == "header":
            if boring_files is None:
                LOG.warning("header strip requested without boring_files; "
                            "skipping the data-driven half of the P-T5 control")
                cur = before
            else:
                cur = [
                    depersonalise_text(
                        t, header_terms=header_map.get(str(b)), ops=("header",),
                        protect_geology=protect_geology)
                    for t, b in zip(before, boring_files)
                ]
        elif op == "place":
            cur = [depersonalise_text(t, ops=("place",),
                                      protect_geology=protect_geology)
                   for t in before]
        else:  # template
            keys = project_keys
            if keys is None and boring_files is not None:
                keys = _project_keys_for(boring_files, metadata=metadata,
                                         metadata_path=metadata_path)
            if keys is None:
                LOG.warning("template normalisation requested without "
                            "project keys; skipping")
                cur = before
            else:
                tmap = template_terms_by_project(
                    before, keys, min_fraction=template_min_fraction,
                    dedupe_keys=boring_files)
                stats["n_projects_with_template"] = len(tmap)
                stats["n_template_terms"] = int(sum(len(v) for v in tmap.values()))
                stats["template_min_fraction"] = template_min_fraction
                stats["template_examples"] = [
                    t for v in list(tmap.values())[:5] for t in v[:2]][:10]
                cur = [remove_template_terms(t, tmap.get(p))
                       for t, p in zip(before, keys)]
                cur = [normalise_text(t) for t in cur]
        stats["per_op"][op] = _removal_stats(before, cur)

    stats["total"] = _removal_stats([str(t) for t in texts], cur)
    stats["n_empty_out"] = int(sum(len(t) == 0 for t in cur))
    return cur, stats


def _project_keys_for(boring_files: Sequence[str], *,
                      metadata: pd.DataFrame | None = None,
                      metadata_path: Path = METADATA_PARQUET) -> list[object]:
    if metadata is None:
        metadata = pd.read_parquet(metadata_path,
                                   columns=["boring_file", "project_key"])
    if "project_key" not in metadata.columns:
        raise ValueError("template normalisation needs a project_key column; "
                         "pass project_keys= explicitly or supply metadata "
                         "carrying project_key")
    m = metadata.copy()
    m["boring_file"] = m["boring_file"].astype(str)
    lut = dict(zip(m["boring_file"], m["project_key"]))
    return [lut.get(str(b)) for b in boring_files]


def cache_tag(mode: str) -> str:
    """Embedding-cache tag for ``mode``.

    FROZEN for the pre-P-T5 modes -- ``lithology_only`` must keep producing
    ``lithonly_<STRIP_VOCAB_VERSION>`` so existing cached embeddings still hit.
    """
    if mode == "full":
        return "full"
    if mode == "lithology_only":
        return f"lithonly_{STRIP_VOCAB_VERSION}"
    if mode == "hardness_only":
        return f"hardness_{STRIP_VOCAB_VERSION}"
    if mode == DEPERSONALISE_MODE:
        return (f"lithonly_{STRIP_VOCAB_VERSION}"
                f"_deperson_{DEPERSONALISE_VERSION}")
    raise ValueError(f"unknown strip mode {mode!r}; expected one of {STRIP_MODES}")


def apply_strip_mode(
    texts: Sequence[str],
    domain: str,
    mode: str,
    *,
    boring_files: Sequence[str] | None = None,
    project_keys: Sequence[object] | None = None,
    metadata: pd.DataFrame | None = None,
    metadata_path: Path = METADATA_PARQUET,
    ops: Sequence[str] = DEPERSONALISE_OPS,
    template_min_fraction: float = TEMPLATE_MIN_DOC_FRACTION,
) -> tuple[list[str], dict]:
    """Batch entry point: ``(stripped_texts, provenance_stats)``.

    The single place a driver should call. For the three frozen modes this is
    exactly ``[strip_text(t, domain, mode) for t in texts]``; for
    ``lithology_only_depersonalised`` it additionally assembles the
    per-borehole header context and the per-project template context.
    """
    if mode not in STRIP_MODES:
        raise ValueError(f"unknown strip mode {mode!r}; expected one of {STRIP_MODES}")
    base = [strip_text(t, domain, "lithology_only") if mode == DEPERSONALISE_MODE
            else strip_text(t, domain, mode) for t in texts]
    if mode != DEPERSONALISE_MODE:
        return base, {"mode": mode, "version": STRIP_VOCAB_VERSION,
                      "cache_tag": cache_tag(mode),
                      **_removal_stats([str(t) for t in texts], base)}
    out, stats = depersonalise_corpus(
        base, boring_files=boring_files, project_keys=project_keys,
        metadata=metadata, metadata_path=metadata_path, ops=ops,
        template_min_fraction=template_min_fraction)
    stats["cache_tag"] = cache_tag(mode)
    stats["lithology_strip"] = _removal_stats([str(t) for t in texts], base)
    stats["lithology_strip"]["version"] = STRIP_VOCAB_VERSION
    return out, stats


def _dictionary_features(texts: list[str], df: pd.DataFrame, domain: str) -> np.ndarray:
    """Rule-based lithology one-hot, no LM. Japan: regime_code + aist_litho_macro_code
    one-hot (if present); else keyword one-hot. UK: BS5930 lithology-keyword one-hot."""
    if domain == "japan" and {"regime_code"}.issubset(df.columns):
        cols = []
        rc = pd.get_dummies(df["regime_code"].astype(int), prefix="rg")
        cols.append(rc.to_numpy(np.float32))
        if "aist_litho_macro_code" in df.columns:
            cols.append(pd.get_dummies(df["aist_litho_macro_code"].astype(int), prefix="lt").to_numpy(np.float32))
        # plus Japanese lithology keyword presence
        kw = ["砂", "礫", "粘土", "シルト", "火山灰", "軽石", "ローム", "岩", "泥", "有機質"]
        kwarr = np.stack([[1.0 if k in t else 0.0 for k in kw] for t in texts]).astype(np.float32)
        cols.append(kwarr)
        return np.concatenate(cols, axis=1)
    # UK: lithology-keyword one-hot
    kw = ["CLAY", "SAND", "GRAVEL", "SILT", "SANDSTONE", "MUDSTONE", "LIMESTONE",
          "CHALK", "PEAT", "TILL", "MADE GROUND"]
    return np.stack([[1.0 if k.lower() in t.lower() else 0.0 for k in kw] for t in texts]).astype(np.float32)


def _tfidf_features(texts: list[str], domain: str) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    # char n-grams: language-agnostic (works for Japanese without a tokenizer)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=400)
    return v.fit_transform(texts).toarray().astype(np.float32)


# --- Structured layer-lithology parser baseline (reviewer ask: is the free text
#     merely a structured lithology label?). A STRONGER rule-based parser than the
#     coarse _dictionary_features: grain size, lithology class, weathering, water
#     state, angularity, colour, and composition %. It captures lithology DETAIL
#     ONLY and deliberately excludes strength/consistency/N vocabulary, so it is the
#     fair comparator -- if the frozen-LM embedding still beats it under LRO, the
#     transferable signal is finer-grained narrative content than a parser recovers;
#     if they tie, "structuring the text transfers" (still a useful finding). ---
_JP_GRAIN = ["粗砂", "中砂", "細砂", "粗礫", "中礫", "細礫", "シルト質", "砂質", "礫質", "粘土質"]
_JP_SORTING = ["淘汰", "均一", "均質", "不均一", "不均質", "分級"]
_JP_LITHCLASS = ["砂", "礫", "粘土", "シルト", "火山灰", "軽石", "ローム", "泥", "岩",
                 "有機質", "腐植", "貝殻", "凝灰", "花崗", "玄武", "安山", "石灰"]
_JP_WEATHER = ["風化", "新鮮", "酸化"]
_JP_WATER = ["含水", "飽和", "乾燥", "湿潤"]
_JP_ANGULAR = ["亜円", "亜角", "円礫", "角礫"]
_JP_COLOR = ["褐", "灰", "黒", "青", "緑", "赤", "黄", "白"]
_UK_GRAIN = ["fine", "medium", "coarse"]
_UK_SORTING = ["well sorted", "poorly sorted", "well graded", "poorly graded",
               "uniformly graded", "gap graded"]
_UK_LITHCLASS = ["clay", "sand", "gravel", "silt", "sandstone", "mudstone", "siltstone",
                 "claystone", "limestone", "dolomite", "chalk", "peat", "marl", "till",
                 "topsoil", "made ground", "cobbles", "boulders"]
_UK_SECONDARY = ["slightly", "very", "gravelly", "sandy", "silty", "clayey"]
_UK_WEATHER = ["weathered", "fresh", "oxidised", "oxidized"]
_UK_ANGULAR = ["angular", "subangular", "sub-angular", "rounded", "subrounded", "sub-rounded"]
_UK_COLOR = ["brown", "grey", "gray", "black", "blue", "green", "red", "yellow", "orange", "white"]
_PCT_RE = re.compile(r"(\d{1,3})\s*[%％]")


def _kw_onehot(texts: list[str], kws: list[str], lower: bool = False) -> np.ndarray:
    if lower:
        return np.stack([[1.0 if k in t.lower() else 0.0 for k in kws] for t in texts]).astype(np.float32)
    return np.stack([[1.0 if k in t else 0.0 for k in kws] for t in texts]).astype(np.float32)


def _pct_bins(text: str) -> list[float]:
    vals = [int(m) for m in _PCT_RE.findall(text)]
    v = max(vals) if vals else -1
    return [1.0 if 0 <= v < 10 else 0.0, 1.0 if 10 <= v < 30 else 0.0,
            1.0 if 30 <= v < 60 else 0.0, 1.0 if v >= 60 else 0.0]


def structured_families(texts: list[str], df: pd.DataFrame, domain: str,
                        *, include_archive_codes: bool = True,
                        ) -> dict[str, np.ndarray]:
    """Per-family blocks of the structured lithology parser.

    ``include_archive_codes=False`` drops the regime/litho-macro one-hots
    from the Japan dictionary block, leaving ONLY text-derived features --
    required whenever the parser is quoted as a language-model-free reading
    of the description itself (the archive codes are map metadata, not
    text). The ``sorting`` family is new (2026-08-12): 淘汰/分級 (JP) and
    graded/sorted (UK) vocabulary for the descriptor-mechanism analysis.
    """
    fams: dict[str, np.ndarray] = {}
    if include_archive_codes:
        fams["dictionary"] = _dictionary_features(texts, df, domain)
    else:
        if domain == "japan":
            kw = ["砂", "礫", "粘土", "シルト", "火山灰", "軽石", "ローム", "岩",
                  "泥", "有機質"]
            fams["dictionary"] = _kw_onehot(texts, kw)
        else:
            fams["dictionary"] = _dictionary_features(texts, df, domain)
    if domain == "japan":
        fams["grain_size"] = _kw_onehot(texts, _JP_GRAIN)
        fams["sorting"] = _kw_onehot(texts, _JP_SORTING)
        fams["lith_class"] = _kw_onehot(texts, _JP_LITHCLASS)
        fams["weathering"] = _kw_onehot(texts, _JP_WEATHER)
        fams["water_state"] = _kw_onehot(texts, _JP_WATER)
        fams["angularity"] = _kw_onehot(texts, _JP_ANGULAR)
        fams["colour"] = _kw_onehot(texts, _JP_COLOR)
    else:
        fams["grain_size"] = _kw_onehot(texts, _UK_GRAIN, lower=True)
        fams["sorting"] = _kw_onehot(texts, _UK_SORTING, lower=True)
        fams["lith_class"] = _kw_onehot(texts, _UK_LITHCLASS, lower=True)
        fams["secondary"] = _kw_onehot(texts, _UK_SECONDARY, lower=True)
        fams["weathering"] = _kw_onehot(texts, _UK_WEATHER, lower=True)
        fams["angularity"] = _kw_onehot(texts, _UK_ANGULAR, lower=True)
        fams["colour"] = _kw_onehot(texts, _UK_COLOR, lower=True)
    fams["composition_pct"] = np.stack(
        [_pct_bins(t) for t in texts]).astype(np.float32)
    return fams


def _structured_litho_features(texts: list[str], df: pd.DataFrame, domain: str) -> np.ndarray:
    """Strong rule-based layer-lithology parser feature matrix (lithology detail only,
    no strength/N terms). Plugs into ``_evaluate_lro`` like any feature block.

    NOTE: for Japan this includes the AIST regime/litho-macro one-hots (archive
    metadata, not text); pass ``include_archive_codes=False`` to
    :func:`structured_families` for the text-derived-only variant. The
    ``sorting`` family postdates the published parser numbers and is NOT
    concatenated here, preserving bit-compatibility of the legacy rung.
    """
    fams = structured_families(texts, df, domain, include_archive_codes=True)
    legacy_order = [k for k in ("dictionary", "grain_size", "lith_class",
                                "secondary", "weathering", "water_state",
                                "angularity", "colour", "composition_pct")
                    if k in fams]
    return np.concatenate([fams[k] for k in legacy_order], axis=1)


def load_domain(domain: str, cache_dir: Path, *,
                per_region_files: int = 500,
                sample_seed: int = 42) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (df with region+n_value+text+base+boring_file cols, base_cols, ignored).

    Every returned frame carries a ``boring_file`` column -- the grouping
    unit for borehole-block permutation nulls and block bootstraps (prereg
    P-T1/T3). Japan: the XML basename; UK: the AGS ``loca_id``.

    ``per_region_files`` / ``sample_seed`` control the Japan subsample
    (defaults reproduce the historical fixed draw: 500 files/region, seed
    42). ``per_region_files=0`` loads the full text-bearing population.
    """
    if domain == "uk":
        df = pd.read_parquet(REPO / "data/features/uk_bgs_spt_full.parquet")
        df = df[(df.n_value > 0) & (df.n_value <= 100)].copy()
        df["text"] = df["lith_desc"].fillna("").astype(str)
        df = df[df.text.str.len() > 0]
        counts = df.region.value_counts()
        df = df[df.region.isin(counts[counts >= 200].index)].reset_index(drop=True)
        df["boring_file"] = df["loca_id"].astype(str)
        base = [c for c in ["depth_from_surface", "ground_level", "latitude_deg", "longitude_deg"] if c in df.columns]
        return df, base, []
    # japan: reuse the leak-proof reconstruction (cached parquet when present,
    # so cluster runs need no XML corpus)
    from scripts.japan_transfer_test import build_dataset
    tag = "full" if per_region_files <= 0 else str(per_region_files)
    cache = cache_dir / f"japan_dataset_{tag}_seed{sample_seed}.parquet"
    df = build_dataset(REPO / "data/features/derived/soil_text_layers.csv",
                       per_region_files=per_region_files, seed=sample_seed,
                       cache=cache)
    base = ["depth_from_surface", "latitude_deg", "longitude_deg"]
    return df, base, []


def _content(per: dict) -> dict:
    regions = sorted(per.keys())
    nt = {r: per[r]["no_text"][0] for r in regions}
    tx = {r: per[r]["text"][0] for r in regions}
    sh = {r: per[r]["shuffled"][0] for r in regions}
    cr = {r: 100 * (tx[r] - sh[r]) / sh[r] for r in regions}
    mean = lambda d: round(float(np.mean(list(d.values()))), 3)
    diffs = [tx[r] - sh[r] for r in regions]
    n_neg = sum(d < 0 for d in diffs)
    from math import comb
    sign_p = sum(comb(len(diffs), k) for k in range(n_neg, len(diffs) + 1)) / 2 ** len(diffs)
    return {"no_text": mean(nt), "shuffled": mean(sh), "text": mean(tx),
            "content_pct": round(100 * (mean(tx) - mean(sh)) / mean(sh), 1),
            "per_region_content": {r: round(cr[r], 1) for r in regions},
            "n_neg": f"{n_neg}/{len(diffs)}", "sign_p": round(sign_p, 5)}


def run(domain: str, out: Path, cache_dir: Path, seeds: list[int] | None = None) -> dict:
    seeds = seeds or [42, 43, 44, 45, 46]
    df, base, _ = load_domain(domain, cache_dir)
    texts_full = df["text"].tolist()
    LOG.info("%s: %d rows, %d regions, base=%s", domain, len(df), df.region.nunique(), base)
    results = {"config": {"domain": domain, "n_rows": len(df), "n_regions": int(df.region.nunique()),
                          "base": base, "seeds": seeds}, "variants": {}}

    # LM embedding variants (full / lithology_only / hardness_only)
    for mode in ("full", "lithology_only", "hardness_only"):
        texts = [strip_text(t, domain, mode) for t in texts_full]
        # report how much was removed
        removed = float(np.mean([1.0 - len(s) / max(len(t), 1) for s, t in zip(texts, texts_full)]))
        ver = "" if mode == "full" else f"_{STRIP_VOCAB_VERSION}"
        emb = embed_texts(texts, cache_dir / f"leak_{domain}_{mode}{ver}_e5.npy")
        per = _evaluate_lro(df, base, emb, seeds)
        c = _content(per); c["mean_frac_chars_removed"] = round(removed, 3)
        results["variants"][f"lm_{mode}"] = c
        LOG.info("lm_%s content %.1f%% (%s, p=%s) | removed %.0f%% chars",
                 mode, c["content_pct"], c["n_neg"], c["sign_p"], 100 * removed)

    # Non-LM baselines
    emb_dict = _dictionary_features(texts_full, df, domain)
    results["variants"]["dictionary_onehot"] = _content(_evaluate_lro(df, base, emb_dict, seeds))
    LOG.info("dictionary content %.1f%% (%s)", results["variants"]["dictionary_onehot"]["content_pct"],
             results["variants"]["dictionary_onehot"]["n_neg"])
    emb_tfidf = _tfidf_features(texts_full, domain)
    results["variants"]["tfidf_char"] = _content(_evaluate_lro(df, base, emb_tfidf, seeds))
    LOG.info("tfidf content %.1f%% (%s)", results["variants"]["tfidf_char"]["content_pct"],
             results["variants"]["tfidf_char"]["n_neg"])
    emb_struct = _structured_litho_features(texts_full, df, domain)
    results["variants"]["structured_litho"] = _content(_evaluate_lro(df, base, emb_struct, seeds))
    LOG.info("structured_litho content %.1f%% (%s) [dim=%d]",
             results["variants"]["structured_litho"]["content_pct"],
             results["variants"]["structured_litho"]["n_neg"], emb_struct.shape[1])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    LOG.info("WROTE %s", out)
    print(json.dumps({k: v.get("content_pct") if isinstance(v, dict) else v
                      for k, v in results["variants"].items()}, indent=2))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", required=True, choices=["japan", "uk"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(a.domain, a.out, a.cache_dir, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
