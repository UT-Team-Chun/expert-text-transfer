"""Unit tests for the structured-lithology parser baseline added for the CEE
round-2 revision (text_leakage_controls), plus the P-T5 depersonalising strip.

Synthetic only, with two exceptions at the bottom that read a small sample of
the real corpus when it is present (skipped otherwise): the P-T5 control is
about what happens to REAL layer descriptions, and a strip that empties a large
share of them would be a deletion masquerading as a clean null.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.text_leakage_controls import (
    DEPERSONALISE_MODE,
    DEPERSONALISE_VERSION,
    STRIP_MODES,
    STRIP_VOCAB_VERSION,
    TEMPLATE_MIN_DOC_FRACTION,
    _PLACE_SUFFIX_MULTI,
    _PLACE_SUFFIX_REJECTED,
    _PLACE_SUFFIX_SINGLE,
    _dictionary_features,
    _kw_onehot,
    _pct_bins,
    _structured_litho_features,
    apply_strip_mode,
    cache_tag,
    depersonalise_corpus,
    depersonalise_text,
    frequent_substrings,
    header_terms_by_borehole,
    header_terms_for_fields,
    normalise_text,
    strip_header_terms,
    strip_place_names,
    strip_text,
    template_terms_by_project,
)


def test_pct_bins_buckets():
    assert _pct_bins("シルト混じり砂を50%含む") == [0.0, 0.0, 1.0, 0.0]  # 50 -> [30,60)
    assert _pct_bins("gravel 5 %") == [1.0, 0.0, 0.0, 0.0]              # 5 -> [0,10)
    assert _pct_bins("80％ sand") == [0.0, 0.0, 0.0, 1.0]               # full-width %, >=60
    assert _pct_bins("no percentage here") == [0.0, 0.0, 0.0, 0.0]      # none -> all zero


def test_kw_onehot_presence_and_case():
    texts = ["fine to coarse SAND", "stiff CLAY"]
    out = _kw_onehot(texts, ["sand", "clay", "gravel"], lower=True)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [1.0, 0.0, 0.0]   # SAND matched case-insensitively
    assert out[1].tolist() == [0.0, 1.0, 0.0]
    # without lower=, exact (here CJK) match
    jp = _kw_onehot(["細砂と礫", "粘土"], ["砂", "礫", "粘土"])
    assert jp[0].tolist() == [1.0, 1.0, 0.0]
    assert jp[1].tolist() == [0.0, 0.0, 1.0]


def test_structured_litho_richer_than_dictionary_uk():
    texts = ["Soft brown slightly gravelly silty CLAY, subangular fine gravel, 20% sand",
             "Dense fine to coarse SAND"]
    df = pd.DataFrame({"region": ["a", "b"]})
    struct = _structured_litho_features(texts, df, "uk")
    dic = _dictionary_features(texts, df, "uk")
    assert struct.shape[0] == 2
    # parser is strictly wider than the coarse dictionary (adds grain/weather/water/...)
    assert struct.shape[1] > dic.shape[1]
    assert np.isfinite(struct).all()
    # row 0 mentions CLAY/gravel/sand + subangular + colour + a percentage -> non-trivial
    assert struct[0].sum() > struct[1].sum()


def test_structured_litho_japan_uses_regime_columns():
    texts = ["褐色の細砂、亜角礫を30%含む、風化", "灰色の粘土、含水"]
    df = pd.DataFrame({"region": ["kanto", "tohoku"],
                       "regime_code": [0, 3],
                       "aist_litho_macro_code": [0, 2]})
    struct = _structured_litho_features(texts, df, "japan")
    assert struct.shape[0] == 2 and struct.shape[1] > 10
    assert np.isfinite(struct).all()


def test_structured_litho_excludes_strength_terms():
    """The parser must NOT key on strength/consistency words (those are the leakage);
    two texts identical except for a strength word must yield identical features."""
    base = "褐色の細砂、礫を20%含む"
    df = pd.DataFrame({"region": ["a", "a"], "regime_code": [0, 0],
                       "aist_litho_macro_code": [0, 0]})
    a = _structured_litho_features([base, base + "、非常に緩い"], df, "japan")
    assert np.array_equal(a[0], a[1])  # adding "非常に緩い" (very loose) changes nothing


# =================================================================================
# P-T5: proper-noun strip + template normalisation
# =================================================================================

REPO = Path(__file__).resolve().parents[3]

# --- the frozen modes ------------------------------------------------------------
# Byte-identical pins. STRIP_VOCAB_VERSION and the stripped text itself are part
# of the embedding cache keys of every completed analysis, so an edit that
# changes any of these outputs silently invalidates cached embeddings (or worse,
# silently reuses embeddings of DIFFERENT text). If one of these fails, the
# change is wrong -- add a new mode instead.
_FROZEN_JP_IN = [
    "非常に緩い細砂、N値=3、含水大",
    "φ2〜5mmの亜角礫を50％含むシルト混じり砂。\\n締まりは中位。",
    "火山灰質の細砂主体、軽石を混入、硬質",
    "瀬戸川層群の砂岩・泥岩互層である。指圧で崩せる。",
    "第三紀島尻層群泥岩新鮮部、貝殻片混入",
]
_FROZEN_JP_LITHOLOGY_ONLY = [
    " い細砂、 = 、 ",
    "φ 〜 mmの亜角礫を ％含むシルト混じり砂。\\n は 。",
    "火山灰質の細砂主体、軽石を混入、 ",
    "瀬戸川層群の砂岩・泥岩互層である。 で崩せる。",
    "第三紀島尻層群泥岩新鮮部、貝殻片混入",
]
_FROZEN_JP_HARDNESS_ONLY = [
    "非常に緩い 、N値=3、含水大",
    "φ2〜5mmの亜角 を50％含む 混じり 。\\n締まりは中位。",
    " 質の 主体、 を混入、硬質",
    "瀬戸川層群の ・ 互層である。指圧で崩せる。",
    "第三紀島尻層群 新鮮部、 片混入",
]
_FROZEN_UK_IN = [
    "Very loose brown fine SAND, N=4",
    "Stiff grey slightly gravelly silty CLAY, 20% sand",
    "MADE GROUND: cobbles and boulders, refusal at 3.5m",
]
_FROZEN_UK_LITHOLOGY_ONLY = [
    "  brown fine SAND,  ",
    "  grey slightly gravelly silty CLAY,  % sand",
    "MADE GROUND: cobbles and boulders,   at  ",
]


def test_frozen_modes_are_byte_identical() -> None:
    assert [strip_text(t, "japan", "full") for t in _FROZEN_JP_IN] == _FROZEN_JP_IN
    assert ([strip_text(t, "japan", "lithology_only") for t in _FROZEN_JP_IN]
            == _FROZEN_JP_LITHOLOGY_ONLY)
    assert ([strip_text(t, "japan", "hardness_only") for t in _FROZEN_JP_IN]
            == _FROZEN_JP_HARDNESS_ONLY)
    assert ([strip_text(t, "uk", "lithology_only") for t in _FROZEN_UK_IN]
            == _FROZEN_UK_LITHOLOGY_ONLY)


def test_frozen_modes_ignore_the_new_context_arguments() -> None:
    """Passing P-T5 context must not perturb a frozen mode's output."""
    terms = header_terms_for_fields(["瀬戸川層群 地質調査"])
    for mode in ("full", "lithology_only", "hardness_only"):
        assert ([strip_text(t, "japan", mode, header_terms=terms,
                            template_terms=["砂岩・泥岩互層"]) for t in _FROZEN_JP_IN]
                == [strip_text(t, "japan", mode) for t in _FROZEN_JP_IN])


def test_cache_tags_are_frozen_and_the_new_mode_is_separate() -> None:
    assert STRIP_VOCAB_VERSION == "v2"          # frozen: participates in cache keys
    assert cache_tag("lithology_only") == "lithonly_v2"
    assert cache_tag("hardness_only") == "hardness_v2"
    assert cache_tag("full") == "full"
    # the new mode carries its OWN version constant, appended -- never a bump of
    # the shared one
    tag = cache_tag(DEPERSONALISE_MODE)
    assert tag == f"lithonly_{STRIP_VOCAB_VERSION}_deperson_{DEPERSONALISE_VERSION}"
    assert len({cache_tag(m) for m in STRIP_MODES}) == len(STRIP_MODES)


def test_unknown_strip_mode_is_an_error() -> None:
    with pytest.raises(ValueError):
        cache_tag("lithology_only_v3")
    with pytest.raises(ValueError):
        apply_strip_mode(["砂"], "japan", "nonsense")


# --- geology must survive ---------------------------------------------------------
# Cases chosen by scanning 300k real layer texts from
# data/features/derived/soil_text_layers.csv for every suffix character the rule
# uses: 山 is 80% 火山*/安山*, 川 is 51% 河川/河道, 区 is 90% 区分, 沢 is 84% 光沢,
# 丘 is 92% 段丘. A strip that ate these would look like a clean null.
_MUST_SURVIVE = [
    "火山灰",                    # the signal the paper is about
    "河川堆積物",
    "山砂利",
    "川砂",
    "細粒火山灰を主体とする",
    "硬質な安山岩、風化が著しい",
    "火山灰質の細砂主体、軽石を混入",
    "岩級区分はCM級である",
    "金属光沢を示す",
    "鏡肌・条線がみられる",
    "低位段丘堆積物",
    "旧河道の埋積土",
    "アスファルト舗装道路",
    "砂丘砂",
    "埋積谷の谷底低地",
    "地山は良好である",
    "桟橋部の玉石混じり砂礫",
    "湖成粘土",
    "三郡変成岩の風化部",
    "国頭層郡の礫岩",
    "第三紀島尻層群泥岩新鮮部",
    "砂岩・泥岩互層",
    "淘汰不良で分級が悪い",
    "山砂を用いた盛土",
]


@pytest.mark.parametrize("text", _MUST_SURVIVE)
def test_geological_vocabulary_survives_place_strip(text: str) -> None:
    assert strip_place_names(text) == text


@pytest.mark.parametrize("text", _MUST_SURVIVE)
def test_geological_vocabulary_survives_the_full_depersonalisation(text: str) -> None:
    """No header/template context: the context-free half must be inert on geology."""
    assert depersonalise_text(text) == normalise_text(text)


def test_place_strip_never_eats_the_volcanic_ash_signal() -> None:
    """The single most dangerous failure: 火山灰 -> 灰 via the 山 suffix rule."""
    for t in ["火山灰", "細粒火山灰", "褐色火山灰質シルト", "火山灰質細砂〜中砂",
              "降下火山灰層", "火山礫を混入する火山灰"]:
        assert strip_place_names(t) == t
        assert "火山灰" in depersonalise_text(t)


def test_geological_words_survive_header_strip_even_when_in_the_header() -> None:
    """The data-driven rule is masked by the protected geological vocabulary:
    a project called 「砂利採取場地質調査」 must not delete 砂利 from the layers."""
    terms = header_terms_for_fields(["砂利採取場地質調査", "砂岩層ボーリング"])
    text = "φ2〜5mmの砂利を混入する砂岩層"
    out = strip_header_terms(text, terms)
    assert "砂利" in out and "砂岩" in out


def test_header_strip_does_not_split_katakana_or_hiragana_words() -> None:
    """Observed on real data before the word-boundary guard: a contractor named
    「大地コンサルタント株式会社」 turned コンクリート into ' クリート' and
    チャートからなる into 'チャート なる'."""
    terms = header_terms_for_fields(["大地コンサルタント株式会社",
                                     "からまつ地区地質調査"])
    text = "コンクリート片を混入する。礫種はチャートからなる。マトリックスは細砂。"
    out = strip_header_terms(text, terms)
    for word in ("コンクリート", "チャート", "からなる", "マトリックス"):
        assert word in out, f"{word} was split by a header n-gram: {out!r}"


# --- identity must be removed ------------------------------------------------------


def _meta_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "boring_file": ["a.html", "b.html"],
        "survey_name": ["常磐線改良工事に伴う地質調査", "美国漁港　地質調査業務"],
        "project_name": [None, None],
        "orderer_name": ["国土交通省関東地方整備局", "国土交通省北海道開発局"],
        "surveyor_name": ["大地コンサルタント株式会社", "株式会社日本地下技術"],
        "project_key": ["常磐線改良工事に伴う地質調査", "美国漁港　地質調査業務"],
    })


def test_header_strip_removes_the_project_name_from_the_layer_text() -> None:
    terms = header_terms_by_borehole(["a.html"], metadata=_meta_frame())["a.html"]
    text = "常磐線改良工事に伴う地質調査の測点、細砂主体で礫を混入する"
    out = strip_header_terms(text, terms)
    assert "常磐線改良工事" not in out
    assert "地質調査" not in out
    assert "細砂" in out and "礫" in out


def test_header_strip_removes_orderer_and_contractor_names() -> None:
    terms = header_terms_by_borehole(["a.html"], metadata=_meta_frame())["a.html"]
    out = strip_header_terms("国土交通省関東地方整備局の管理する堤防、"
                             "大地コンサルタント株式会社が実施、粘性土", terms)
    assert "関東地方整備局" not in out
    assert "コンサルタント" not in out
    assert "粘性土" in out and "堤防" in out


def test_header_strip_is_per_borehole() -> None:
    """b.html's header must not strip a.html's project name."""
    m = header_terms_by_borehole(["a.html", "b.html"], metadata=_meta_frame())
    text = "常磐線改良工事に伴う地質調査、細砂"
    assert "常磐線" not in strip_header_terms(text, m["a.html"])
    assert "常磐線" in strip_header_terms(text, m["b.html"])


@pytest.mark.parametrize("stem,suffix", [
    ("東京", "都"), ("大阪", "府"), ("千葉", "県"), ("横浜", "市"), ("軽井", "町"),
    ("白川", "村"), ("多古", "郡"), ("多摩", "川"), ("富士", "山"), ("奄美", "島"),
    ("支笏", "湖"), ("鹿児", "港"), ("東神奈", "駅"), ("永代", "橋"),
    ("第二", "工区"), ("本村", "地区"), ("光ヶ", "団地"), ("松原", "地内"),
    ("川崎", "地先"),
])
def test_every_supported_place_suffix_is_stripped(stem: str, suffix: str) -> None:
    text = f"{stem}{suffix}の細砂、礫を混入する"
    out = strip_place_names(text)
    assert stem not in out, f"stem survived: {out!r}"
    assert suffix not in out, f"suffix survived: {out!r}"
    assert "細砂" in out and "礫" in out


def test_supported_suffix_lists_cover_the_prereg_wording() -> None:
    for ch in "都府県市町村郡川山島湖港駅橋":
        assert ch in _PLACE_SUFFIX_SINGLE
    assert "工区" in _PLACE_SUFFIX_MULTI and "地区" in _PLACE_SUFFIX_MULTI


@pytest.mark.parametrize("suffix,text", [
    ("道", "アスファルト舗装道路の直下"),
    ("道", "旧河道の埋積土"),
    ("線", "鏡肌・条線がみられる"),
    ("区", "岩級区分はCM級"),
    ("沢", "絹糸光沢を示す"),
    ("丘", "低位段丘堆積物"),
    ("谷", "埋積谷の谷底低地"),
    ("台", "橋台背面の盛土"),
])
def test_rejected_suffixes_are_documented_and_inert(suffix: str, text: str) -> None:
    """Each rejected suffix carries its corpus-measured reason, and the
    geological term it would have damaged is untouched."""
    assert suffix in _PLACE_SUFFIX_REJECTED
    assert _PLACE_SUFFIX_REJECTED[suffix]
    assert suffix not in _PLACE_SUFFIX_SINGLE
    assert strip_place_names(text) == text


def test_one_character_stems_are_left_alone() -> None:
    """PLACE_MIN_STEM=2 is what keeps 火|山 and 河|川 out of the rule's reach."""
    assert strip_place_names("火山灰と河川堆積物") == "火山灰と河川堆積物"
    assert strip_place_names("山砂利、川砂") == "山砂利、川砂"


# --- template normalisation --------------------------------------------------------


def test_template_removes_90pct_boilerplate_but_keeps_a_30pct_string() -> None:
    boiler = "本調査は当該事業の一環として実施した"   # in 9/10 layers
    minority = "貝殻片を多量に混入する"               # in 3/10 layers
    # Layer descriptions VARY, as they do in a real borehole log -- see
    # test_template_deletes_project_constant_text_by_design for what happens
    # when they do not.
    liths = ["細砂主体", "中砂を挟む", "粘性土", "砂礫層", "シルト混じり砂",
             "火山灰質シルト", "軽石を混入", "腐植土", "礫混じり砂", "凝灰岩"]
    texts = []
    for i, lith in enumerate(liths):
        parts = []
        if i < 9:
            parts.append(boiler)
        parts.append(f"{lith}、深度{i}m")
        if i < 3:
            parts.append(minority)
        texts.append("。".join(parts))
    keys = ["P1"] * 10
    terms = template_terms_by_project(texts, keys)["P1"]
    assert any(boiler in t for t in terms), terms
    assert not any(minority in t for t in terms), terms
    out, stats = depersonalise_corpus(texts, project_keys=keys,
                                      ops=("normalise", "template"))
    assert all(boiler not in t for t in out)
    assert sum(minority in t for t in out) == 3
    for lith, t in zip(liths, out):
        assert lith in t, f"layer-specific lithology deleted: {t!r}"
    assert stats["per_op"]["template"]["frac_chars_removed"] > 0
    assert stats["n_empty_out"] == 0


def test_template_deletes_project_constant_text_by_design() -> None:
    """Text identical across >=90% of a project's layers is deleted even when it
    is lithological -- and that is correct, not a bug.

    The pre-registration defines the operation on document frequency alone
    ("substrings present in a high fraction of that project's layer texts"),
    and a phrase that never varies within the project carries no
    layer-specific information there. The error is CONSERVATIVE: over-deletion
    can only attenuate the content effect, never inflate it, so P-T5 stays a
    genuine test rather than a rubber stamp. Measured cost on the full
    population is small (0.43% of characters, 2.3% of layers touched, 5 layers
    of 1.3M emptied), so this is not a licence to delete description wholesale.
    """
    texts = [f"φ2〜5mmの亜角礫を混入する。深度{i}m" for i in range(8)]
    out, _ = depersonalise_corpus(texts, project_keys=["P1"] * 8,
                                  ops=("normalise", "template"))
    assert all("亜角礫を混入する" not in t for t in out)
    assert all(t.strip() for t in out)     # but never emptied


def test_template_threshold_is_configurable_and_defaults_to_90pct() -> None:
    assert TEMPLATE_MIN_DOC_FRACTION == 0.9
    texts = [("共通の定型文である。" if i < 5 else "") + f"細砂、深度{i}m"
             for i in range(10)]   # boilerplate in exactly 5/10
    keys = ["P1"] * 10
    assert "P1" not in template_terms_by_project(texts, keys)          # 0.5 < 0.9
    got = template_terms_by_project(texts, keys, min_fraction=0.5)["P1"]
    assert any("共通の定型文" in t for t in got)


def test_template_is_per_project() -> None:
    boiler = "◯◯建設コンサルタント標準様式による"
    texts = [boiler + f"細砂{i}" for i in range(6)] + [boiler + f"粘土{i}"
                                                      for i in range(6)]
    keys = ["P1"] * 6 + ["P2"] * 6
    terms = template_terms_by_project(texts[:6] + ["粘土X"] * 6, keys)
    assert "P1" in terms and "P2" not in terms


def test_template_skips_projects_with_too_few_layers() -> None:
    texts = ["定型文が入っています。細砂"] * 3
    assert template_terms_by_project(texts, ["P1"] * 3) == {}


def test_frequent_substrings_returns_maximal_terms_only() -> None:
    texts = ["ABCDEFGH-1", "ABCDEFGH-2", "ABCDEFGH-3"]
    # "ABCDEFGH-" (hyphen included: it too is in all three) and NOT also its
    # sub-strings ABCDEFGH / ABCDEFG / BCDEFGH / ...
    assert frequent_substrings(texts, min_fraction=1.0, min_len=6) == ["ABCDEFGH-"]


def test_frequent_substrings_needs_the_high_fraction() -> None:
    texts = ["ABCDEFGH-1", "ABCDEFGH-2", "XYZ-3", "XYZ-4"]
    assert frequent_substrings(texts, min_fraction=0.9, min_len=6) == []
    assert frequent_substrings(texts, min_fraction=0.5, min_len=6) == ["ABCDEFGH-"]


# --- composition, independence, idempotence -----------------------------------------


def test_normalisation_collapses_width_and_whitespace_variants() -> None:
    a = normalise_text("礫を５０％含む　　砂\\n細砂")
    b = normalise_text("礫を50%含む 砂 細砂")
    assert a == b == "礫を50%含む 砂 細砂"
    assert normalise_text(a) == a          # idempotent


def test_operations_can_be_applied_independently() -> None:
    text = "常磐線改良工事の多摩川右岸、細砂"
    terms = header_terms_by_borehole(["a.html"], metadata=_meta_frame())["a.html"]
    only_place = depersonalise_text(text, header_terms=terms,
                                    ops=("normalise", "place"))
    only_header = depersonalise_text(text, header_terms=terms,
                                     ops=("normalise", "header"))
    assert "多摩川" not in only_place and "常磐線改良工事" in only_place
    assert "常磐線改良工事" not in only_header and "多摩川" in only_header
    both = depersonalise_text(text, header_terms=terms)
    assert "多摩川" not in both and "常磐線改良工事" not in both
    assert "細砂" in both


def test_unknown_op_is_an_error() -> None:
    with pytest.raises(ValueError):
        depersonalise_text("砂", ops=("normalise", "shuffle"))
    with pytest.raises(ValueError):
        depersonalise_corpus(["砂"], ops=("shuffle",))


def test_depersonalisation_is_idempotent() -> None:
    terms = header_terms_by_borehole(["a.html"], metadata=_meta_frame())["a.html"]
    tmpl = ["本調査は当該事業の一環として実施した"]
    cases = [
        "常磐線改良工事に伴う地質調査、多摩川右岸の細砂、火山灰質",
        "本調査は当該事業の一環として実施した。φ2〜5mmの亜角礫を50％含む",
        "東京都千代田区の盛土　　\\n河川堆積物、山砂利",
        "第三紀島尻層群泥岩新鮮部、貝殻片混入",
    ]
    for c in cases:
        once = depersonalise_text(c, header_terms=terms, template_terms=tmpl)
        assert depersonalise_text(once, header_terms=terms,
                                  template_terms=tmpl) == once


def test_corpus_pipeline_matches_the_single_text_pipeline() -> None:
    meta = _meta_frame()
    texts = ["常磐線改良工事の測点、多摩川右岸の細砂",
             "美国漁港の岸壁、火山灰質シルト"]
    files = ["a.html", "b.html"]
    out, _ = depersonalise_corpus(texts, boring_files=files, metadata=meta,
                                  ops=("normalise", "header", "place"))
    hmap = header_terms_by_borehole(files, metadata=meta)
    expect = [depersonalise_text(t, header_terms=hmap[f],
                                 ops=("normalise", "header", "place"))
              for t, f in zip(texts, files)]
    assert out == expect


def test_apply_strip_mode_composes_lithology_strip_then_depersonalisation() -> None:
    meta = _meta_frame()
    texts = ["常磐線改良工事、非常に緩い細砂、N値=3、多摩川右岸",
             "美国漁港、硬質な火山灰質シルト、打撃回数15"]
    out, stats = apply_strip_mode(texts, "japan", DEPERSONALISE_MODE,
                                  boring_files=["a.html", "b.html"],
                                  metadata=meta)
    joined = " ".join(out)
    assert "常磐線" not in joined and "多摩川" not in joined     # identity gone
    assert "緩い" not in joined and "硬質" not in joined         # strength gone (v2)
    assert "細砂" in joined and "火山灰" in joined               # lithology kept
    assert stats["mode"] == DEPERSONALISE_MODE
    assert stats["version"] == DEPERSONALISE_VERSION
    assert stats["cache_tag"] == cache_tag(DEPERSONALISE_MODE)
    assert set(stats["per_op"]) == {"normalise", "header", "place", "template"}
    assert stats["lithology_strip"]["version"] == STRIP_VOCAB_VERSION
    assert stats["n_empty_out"] == 0
    for op in stats["per_op"].values():
        assert {"frac_chars_removed", "chars_removed", "n_texts_changed",
                "n_emptied"} <= set(op)


def test_apply_strip_mode_is_a_passthrough_for_the_frozen_modes() -> None:
    out, stats = apply_strip_mode(_FROZEN_JP_IN, "japan", "lithology_only")
    assert out == _FROZEN_JP_LITHOLOGY_ONLY
    assert stats["cache_tag"] == "lithonly_v2"


def test_strip_text_reaches_the_new_mode_without_context() -> None:
    """No header/template context available -> the context-free half still runs."""
    got = strip_text("多摩川右岸の非常に緩い細砂、N値=3", "japan", DEPERSONALISE_MODE)
    assert "多摩川" not in got and "緩い" not in got and "細砂" in got


def test_depersonalisation_without_boring_files_skips_the_data_driven_half() -> None:
    out, stats = apply_strip_mode(["常磐線改良工事、細砂"], "japan",
                                  DEPERSONALISE_MODE)
    assert stats["per_op"]["header"]["chars_removed"] == 0
    assert "細砂" in out[0]


# --- wiring into the runner ----------------------------------------------------------


def test_embedding_cache_key_is_per_strip_mode(tmp_path) -> None:
    """A new mode must get its OWN cache entry -- verified, not assumed."""
    from scripts.nc_grouped_null import embedding_cache_path

    litho = [strip_text(t, "japan", "lithology_only") for t in _FROZEN_JP_IN]
    deperson, _ = apply_strip_mode(_FROZEN_JP_IN, "japan", DEPERSONALISE_MODE)
    p_litho = embedding_cache_path(tmp_path, "japan", litho, "lithology_only")
    p_new = embedding_cache_path(tmp_path, "japan", deperson, DEPERSONALISE_MODE)
    assert p_litho != p_new
    assert p_litho.name.startswith("grouped_japan_lithonly_v2_")   # frozen format
    assert "deperson_" in p_new.name
    # the hash differs too, so even a same-named cache could not collide
    assert p_litho.name.split("_")[-2] != p_new.name.split("_")[-2]


def test_runner_defaults_to_the_frozen_mode(monkeypatch, tmp_path) -> None:
    import scripts.nc_grouped_null as gn

    seen = {}

    def _fake_run(domain, out, cache_dir, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(gn, "run", _fake_run)
    gn.main(["--out", str(tmp_path / "o.json")])
    assert seen["strip_mode"] == "lithology_only"
    gn.main(["--out", str(tmp_path / "o.json"), "--strip-mode", DEPERSONALISE_MODE])
    assert seen["strip_mode"] == DEPERSONALISE_MODE


def test_cli_arguments_bind_to_the_real_run_signature(monkeypatch, tmp_path) -> None:
    """Every kwarg main() sends must exist on run().

    Regression guard for a concurrent-edit hazard, not a hypothetical: two
    sessions editing the argparse block and the run() signature at once can
    each keep half of the other's change, leaving main() passing a keyword
    run() does not accept -- or run()'s body reading a name that is no longer
    a parameter. Neither shows up in the mechanics tests, which call
    evaluate_grouped directly.
    """
    import inspect

    import scripts.nc_grouped_null as gn

    real_run = gn.run
    seen = {}
    monkeypatch.setattr(gn, "run", lambda *a, **kw: seen.update(kw) or {})
    gn.main(["--out", str(tmp_path / "o.json")])
    # raises TypeError if main() passes something run() cannot accept
    inspect.signature(real_run).bind("japan", tmp_path / "o.json", tmp_path, **seen)
    assert {"strip_mode", "strata_col"} <= set(seen)


# --- real-corpus sanity: is this a control or a deletion? ------------------------------

_LAYERS_CSV = REPO / "data/features/derived/soil_text_layers.csv"
_META_PARQUET = REPO / "data/features/derived/kunijiban_metadata.parquet"


@pytest.mark.skipif(not (_LAYERS_CSV.exists() and _META_PARQUET.exists()),
                    reason="real corpus not available")
def test_real_corpus_sample_is_a_control_not_a_deletion() -> None:
    lay = pd.read_csv(_LAYERS_CSV, usecols=["file_path", "observation_text"],
                      nrows=3000)
    lay = lay[lay.observation_text.fillna("").str.len() > 0]
    texts = lay.observation_text.astype(str).tolist()
    files = [Path(p).name for p in lay.file_path]
    litho = [strip_text(t, "japan", "lithology_only") for t in texts]
    out, stats = apply_strip_mode(texts, "japan", DEPERSONALISE_MODE,
                                  boring_files=files)
    n = len(out)
    # already-empty inputs do not count against the strip
    newly_empty = sum(len(a) == 0 and len(b.strip()) > 0 for a, b in zip(out, litho))
    assert newly_empty / n < 0.01, f"{newly_empty}/{n} layers emptied"
    assert stats["total"]["frac_chars_removed"] < 0.25
    # geological vocabulary is essentially untouched (< 1% of occurrences)
    for kw in ("火山灰", "砂", "礫", "粘土", "シルト", "安山", "凝灰", "段丘"):
        before = sum(t.count(kw) for t in litho)
        after = sum(t.count(kw) for t in out)
        if before:
            assert after >= 0.99 * before, f"{kw}: {before} -> {after}"


@pytest.mark.skipif(not (_LAYERS_CSV.exists() and _META_PARQUET.exists()),
                    reason="real corpus not available")
def test_real_corpus_sample_strip_is_idempotent() -> None:
    lay = pd.read_csv(_LAYERS_CSV, usecols=["file_path", "observation_text"],
                      nrows=2000)
    lay = lay[lay.observation_text.fillna("").str.len() > 0]
    texts = [strip_text(str(t), "japan", "lithology_only")
             for t in lay.observation_text]
    files = [Path(p).name for p in lay.file_path]
    once, _ = depersonalise_corpus(texts, boring_files=files)
    twice, _ = depersonalise_corpus(once, boring_files=files)
    assert twice == once
