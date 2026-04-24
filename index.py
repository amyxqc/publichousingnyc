"""
NYT Sentiment Analysis — Chen (2026) LES Public Housing Paper
=============================================================
Section 5.2: Newspaper sentiment as a test of the Husock narrative

EXTENDED VERSION: 1900 → present (2026)
---------------------------------------
The paper's core analysis is 1930–1970, but this run extends the time series
to the full NYT archive available (1900 onward) for long-run context. The
paper-specific structural-break comparison and period shading (I/II/III)
remain anchored to the 1930–1970 window; everything outside that window is
shown without the paper's periodization overlay.

CAVEAT: The domain-adapted VADER lexicon and framing categories were
calibrated for mid-20th-century housing discourse. They still apply
reasonably to pre-1930 tenement-era coverage and post-1970 NYCHA
coverage, but terms specific to later eras (gentrification, Section 8,
RAD, vouchers, etc.) are not in the lexicon. Interpret the extended tails
as directional context rather than a calibrated measurement.

SETUP (run once in terminal):
    pip install requests vaderSentiment pandas matplotlib scipy tqdm

USAGE:
    1. Set NYT_API_KEY below (get free key at developer.nytimes.com)
    2. Run:  python nyt_sentiment.py
    3. Outputs written to ./sentiment_output/

The script has three modes:
    MODE = "collect"   → hit the NYT Archive API and save raw CSV
    MODE = "analyze"   → load saved CSV and run all analysis
    MODE = "both"      → collect then analyze in one run

Set MODE below before running. On first run use "both".
If the API call is interrupted, set MODE = "analyze" to re-run
analysis on whatever data was already saved.
"""

# ─── USER SETTINGS ────────────────────────────────────────────────────────────
NYT_API_KEY = "7DdBw359pt8APA0PFXSdRNdoZz1d3eo2ek6AAOI0ak7eYRFz"   # paste your NYT API key here
MODE        = "both"            # "collect" | "analyze" | "both"
OUT_DIR     = "./sentiment_output"

# Time window (extended from paper's 1930–1970)
MIN_YEAR    = 1900
MAX_YEAR    = 2026              # "present" — current year is truncated to last completed month

# Paper-specific window (keep for overlays / break comparison)
PAPER_START = 1930
PAPER_END   = 1970
# ──────────────────────────────────────────────────────────────────────────────

import os, time, json, re, csv, math
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DOMAIN-ADAPTED VADER LEXICON
# Augmentations from Section 5.2 of the paper
# ═══════════════════════════════════════════════════════════════════════════════

# Positive housing-reform terms (VADER internal scale: -4 to +4)
POSITIVE_TERMS = {
    "modern":       2.0,
    "sanitary":     2.5,
    "decent":       2.0,
    "spacious":     2.0,
    "ventilated":   1.5,
    "improved":     1.5,
    "clearance":    1.0,   # switched to -1.5 post-1955 — see scoring fn
    "wholesome":    2.0,
    "adequate":     1.5,
    "rehabilitation": 1.5,
    "revitalization": 1.5,
    "relief":       1.5,
}

# Negative housing-pathology terms
NEGATIVE_TERMS = {
    "slum":         -2.5,
    "slums":        -2.5,
    "blight":       -2.5,
    "blighted":     -2.5,
    "decrepit":     -2.0,
    "vermin":       -2.5,
    "overcrowded":  -2.0,
    "overcrowding": -2.0,
    "dilapidated":  -2.0,
    "squalid":      -3.0,
    "squalor":      -3.0,
    "crime-ridden": -2.5,
    "delinquency":  -1.5,
    "delinquent":   -1.5,
    "gang":         -2.0,
    "gangs":        -2.0,
    "vandalism":    -2.0,
    "deterioration":-2.0,
    "deteriorating":-2.0,
    "abandoned":    -2.0,
    "abandonment":  -2.0,
    "neglect":      -2.0,
    "neglected":    -2.0,
    "rundown":      -2.0,
}

# Semantic-drift terms: score depends on year
# Format: term -> {before_year: score, after_year: score, cutoff: year}
DRIFT_TERMS = {
    "project":   {"before": 0.0,  "after": -1.0, "cutoff": 1950},
    "projects":  {"before": 0.0,  "after": -1.0, "cutoff": 1950},
    "clearance": {"before": 1.0,  "after": -1.5, "cutoff": 1955},
    "welfare":   {"before": 0.5,  "after": -1.0, "cutoff": 1955},
    "troubled":  {"before": -1.0, "after": -1.5, "cutoff": 1960},
}


def build_vader(year):
    """Return VADER analyzer with domain augmentations for a given year."""
    analyzer = SentimentIntensityAnalyzer()
    # Apply static augmentations
    for term, score in POSITIVE_TERMS.items():
        analyzer.lexicon[term] = score
    for term, score in NEGATIVE_TERMS.items():
        analyzer.lexicon[term] = score
    # Apply year-sensitive drift terms
    for term, cfg in DRIFT_TERMS.items():
        score = cfg["before"] if year < cfg["cutoff"] else cfg["after"]
        analyzer.lexicon[term] = score
    return analyzer


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KEYWORD FILTERS  (Section 5.2)
# ═══════════════════════════════════════════════════════════════════════════════

TIER1_TERMS = [
    "public housing",
    "housing project",
    "housing projects",
    "nycha",
    "housing authority",
    "slum clearance",
    "urban renewal",
    "tenement",
    "tenements",
    "first houses",
    "vladeck houses",
    "laguardia houses",
    "la guardia houses",
    "baruch houses",
    "alfred e. smith houses",
    "lillian wald houses",
    "jacob riis houses",
    "wald houses",
    "riis houses",
    "smith houses",
    "gompers houses",
    "rutgers houses",
]

TIER2_GEO = [
    "lower east side",
    "east side",
    "manhattan",
    "new york city",
    "new york",
]

# Broad terms that MUST also match a geo term
BROAD_TERMS = {"tenement", "tenements", "housing authority"}

EXCLUDE_SECTION = {"real estate"}   # section_name values to exclude unless geo match
EXCLUDE_MATERIAL = {
    "letter", "letters", "correction", "corrections",
    "paid notice", "classified", "obituaries"
}


def article_text(article):
    """Concatenate headline + snippet + lead_paragraph."""
    parts = []
    headline = article.get("headline", {})
    if isinstance(headline, dict):
        parts.append(headline.get("main", "") or "")
    elif isinstance(headline, str):
        parts.append(headline)
    parts.append(article.get("snippet", "") or "")
    parts.append(article.get("lead_paragraph", "") or "")
    return " ".join(parts).lower()


def passes_keyword_filter(article, year):
    """Return True if article matches keyword criteria."""
    text = article_text(article)
    material = (article.get("type_of_material") or "").lower()
    section  = (article.get("section_name") or "").lower()

    # Exclude known non-relevant material types
    if any(exc in material for exc in EXCLUDE_MATERIAL):
        return False

    # Must match at least one Tier 1 term
    matched_tier1 = [t for t in TIER1_TERMS if t in text]
    if not matched_tier1:
        return False

    # If only broad terms matched, require a geo term
    non_broad = [t for t in matched_tier1 if t not in BROAD_TERMS]
    if not non_broad:
        if not any(g in text for g in TIER2_GEO):
            return False

    # Exclude real-estate section without LES geo match
    if section in EXCLUDE_SECTION:
        if "lower east side" not in text and "east side" not in text:
            return False

    # Era-specific filters
    # "Public housing" as a term largely post-dates 1933 (Wagner-Steagall lead-up).
    # Before 1933, require tenement/slum coverage rather than "public housing" per se.
    if "public housing" in matched_tier1 and year < 1933:
        # Keep only if also matched another Tier 1 term (tenement, slum clearance, etc.)
        other = [t for t in matched_tier1 if t != "public housing"]
        if not other:
            return False

    # Urban renewal only from 1949 onward
    if "urban renewal" in matched_tier1 and year < 1949:
        return False

    # NYCHA founded 1934 — don't credit NYCHA mentions pre-1934
    if "nycha" in matched_tier1 and year < 1934:
        other = [t for t in matched_tier1 if t != "nycha"]
        if not other:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FRAMING CATEGORIES  (Section 5.2)
# ═══════════════════════════════════════════════════════════════════════════════

FRAMES = {
    "reform":       ["modern", "decent", "sanitary", "relief",
                     "better housing", "replaces slums", "improved",
                     "health", "clean", "wholesome", "light and air",
                     "model housing", "model tenement"],
    "construction": ["units", "construction begins", "approved",
                     "budget", "federal funds", "groundbreaking",
                     "opens", "dedicated", "completed", "contract",
                     "authorized", "appropriation"],
    "conflict":     ["protest", "oppose", "dispute", "segregation",
                     "displaced", "community", "tenant", "strike",
                     "integration", "discrimination", "opposition",
                     "evicted", "relocation"],
    "pathology":    ["crime", "welfare", "deteriorat", "troubled",
                     "gang", "delinquency", "drugs", "vandalism",
                     "violence", "shooting", "robbery", "arrest",
                     "disorder", "unsafe", "dangerous"],
    "crisis":       ["crisis", "failure", "collapse", "demolish",
                     "abandon", "overhaul", "bankrupt", "shortfall",
                     "underfunded", "deteriorated", "condemned",
                     "scandal", "nightmare"],
}


def assign_frame(text):
    """Return dominant frame label based on marker term frequency."""
    scores = {}
    for frame, markers in FRAMES.items():
        scores[frame] = sum(1 for m in markers if m in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "construction"  # default


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def score_article(article, year):
    """Return dict of scores for one article."""
    text  = article_text(article)
    analyzer = build_vader(year)
    vs    = analyzer.polarity_scores(text)

    # Reform and pathology indices (proportion of words matching each list)
    words = text.split()
    n     = max(len(words), 1)
    reform_idx   = sum(1 for w in words
                       if w in POSITIVE_TERMS) / n
    pathology_idx= sum(1 for w in words
                       if w in NEGATIVE_TERMS) / n
    denom = reform_idx + pathology_idx
    frame_ratio  = (reform_idx - pathology_idx) / denom if denom > 0 else 0.0
    frame        = assign_frame(text)

    wc = int(article.get("word_count") or 0)

    return {
        "id":           article.get("_id", ""),
        "pub_date":     article.get("pub_date", "")[:10],
        "year":         year,
        "headline":     (article.get("headline") or {}).get("main", ""),
        "section":      article.get("section_name", ""),
        "material":     article.get("type_of_material", ""),
        "word_count":   wc,
        "compound":     round(vs["compound"], 4),
        "pos":          round(vs["pos"], 4),
        "neg":          round(vs["neg"], 4),
        "neu":          round(vs["neu"], 4),
        "reform_idx":   round(reform_idx, 5),
        "pathology_idx":round(pathology_idx, 5),
        "frame_ratio":  round(frame_ratio, 4),
        "frame":        frame,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA COLLECTION — NYT Archive API
# ═══════════════════════════════════════════════════════════════════════════════

RAW_CSV = os.path.join(OUT_DIR, "nyt_raw_corpus.csv")

FIELDNAMES = [
    "id", "pub_date", "year", "headline", "section", "material",
    "word_count", "compound", "pos", "neg", "neu",
    "reform_idx", "pathology_idx", "frame_ratio", "frame"
]


def months_for_year(year, now):
    """Number of months to request for a given year — truncate current year."""
    if year < now.year:
        return 12
    if year == now.year:
        # Only request months that have completed (current month - 1)
        # Use current month - 1 to avoid partial-month noise; if January, skip.
        return max(0, now.month - 1)
    return 0   # future years


def collect():
    """Hit the NYT Archive API for MIN_YEAR..MAX_YEAR and save filtered corpus."""
    if NYT_API_KEY == "YOUR_KEY_HERE":
        raise ValueError(
            "Set NYT_API_KEY at the top of this script before collecting.\n"
            "Get a free key at: https://developer.nytimes.com/get-started"
        )

    now = datetime.now()

    # Count total API calls for time estimate
    total_calls = sum(months_for_year(y, now)
                      for y in range(MIN_YEAR, MAX_YEAR + 1))
    hours_est   = total_calls * 12 / 3600

    print(f"  Range: {MIN_YEAR}–{MAX_YEAR}")
    print(f"  Total API calls planned: {total_calls}")
    print(f"  Estimated time: ~{hours_est:.1f} hours at 5 req/min rate limit")
    print(f"  (Archive API is slow; grab a coffee or three.)\n")

    seen_ids = set()
    rows     = []
    errors   = []

    # Load already-collected rows if resuming
    if os.path.exists(RAW_CSV):
        existing = pd.read_csv(RAW_CSV)
        seen_ids = set(existing["id"].tolist())
        rows     = existing.to_dict("records")
        print(f"  Resuming: {len(rows)} articles already collected.\n")

    for year in range(MIN_YEAR, MAX_YEAR + 1):
        n_months = months_for_year(year, now)
        for month in range(1, n_months + 1):
            url = (f"https://api.nytimes.com/svc/archive/v1/"
                   f"{year}/{month}.json?api-key={NYT_API_KEY}")
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    print(f"  Rate limited at {year}-{month:02d}, "
                          f"sleeping 60s…")
                    time.sleep(60)
                    resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                articles = data.get("response", {}).get("docs", [])
            except Exception as e:
                print(f"  ERROR {year}-{month:02d}: {e}")
                errors.append(f"{year}-{month:02d}: {e}")
                time.sleep(12)
                continue

            month_count = 0
            for art in articles:
                aid = art.get("_id", "")
                if aid in seen_ids:
                    continue
                if passes_keyword_filter(art, year):
                    row = score_article(art, year)
                    rows.append(row)
                    seen_ids.add(aid)
                    month_count += 1

            print(f"  {year}-{month:02d}: {len(articles):5d} total, "
                  f"{month_count:3d} matched  "
                  f"(corpus total: {len(rows)})")

            # Save incrementally every month
            pd.DataFrame(rows, columns=FIELDNAMES).to_csv(
                RAW_CSV, index=False)

            # Respect 5 req/min for Archive endpoint
            time.sleep(12)

    print(f"\nCollection complete. {len(rows)} articles in corpus.")
    if errors:
        with open(os.path.join(OUT_DIR, "collection_errors.txt"), "w") as f:
            f.write("\n".join(errors))
        print(f"  {len(errors)} errors logged.")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANALYSIS — time series, structural breaks, frame shares
# ═══════════════════════════════════════════════════════════════════════════════

# Palette matching paper figures
NAVY  = '#1a3a5c'
BLUE  = '#2166ac'
MID   = '#4393c3'
RED   = '#c0392b'
AMBER = '#b47a14'
TEAL  = '#006464'
GREY  = '#666666'
LGREY = '#cccccc'

FRAME_COLORS = {
    "reform":       '#2166ac',
    "construction": '#aaaaaa',
    "conflict":     '#e07b39',
    "pathology":    '#c0392b',
    "crisis":       '#7b1a1a',
}

PERIOD_COLORS = {
    'I':   '#1a3a5c',
    'II':  '#2166ac',
    'III': '#4393c3',
}


def period_shade(ax, alpha=0.07):
    """Shade the paper's three periods (1930–1970 only). Rest of timeline unshaded."""
    ax.axvspan(1930, 1941, alpha=alpha, color=PERIOD_COLORS['I'],  zorder=0)
    ax.axvspan(1942, 1959, alpha=alpha, color=PERIOD_COLORS['II'], zorder=0)
    ax.axvspan(1960, 1970, alpha=alpha, color=PERIOD_COLORS['III'],zorder=0)
    for xv in [1941, 1959]:
        ax.axvline(xv, color=GREY, lw=0.9, ls='--', alpha=0.5)
    # Mark paper-window boundaries softly
    for xv in [PAPER_START, PAPER_END]:
        ax.axvline(xv, color=GREY, lw=0.6, ls=':', alpha=0.4)


def bai_perron_breaks(series, max_breaks=3, min_segment=5):
    """
    Simple sequential Bai-Perron implementation.
    Finds up to max_breaks structural breaks in a 1-D series by minimising
    the sum of squared residuals (constant-mean model per segment).
    Returns list of break years (index positions, converted to years by caller).
    """
    n = len(series)
    y = np.array(series)

    def ssr(seg):
        return np.sum((seg - seg.mean())**2) if len(seg) > 0 else 0.0

    def find_one_break(y_sub, min_seg):
        best_ssr = np.inf
        best_bp  = None
        for bp in range(min_seg, len(y_sub) - min_seg + 1):
            s = ssr(y_sub[:bp]) + ssr(y_sub[bp:])
            if s < best_ssr:
                best_ssr = s
                best_bp  = bp
        return best_bp, best_ssr

    # Start with single-segment SSR
    total_ssr0 = ssr(y)
    breaks = []
    segments = [(0, n)]   # (start, end) inclusive of end

    for _ in range(max_breaks):
        best_gain   = 0
        best_new_bp = None
        best_seg_idx= None

        for si, (start, end) in enumerate(segments):
            seg = y[start:end]
            if (end - start) < 2 * min_segment:
                continue
            bp, new_ssr = find_one_break(seg, min_segment)
            if bp is None:
                continue
            old_ssr = ssr(seg)
            gain = old_ssr - new_ssr
            if gain > best_gain:
                best_gain    = gain
                best_new_bp  = start + bp
                best_seg_idx = si

        if best_new_bp is None or best_gain < 1e-6:
            break

        breaks.append(best_new_bp)
        breaks.sort()
        # Rebuild segments
        boundaries = [0] + breaks + [n]
        segments   = [(boundaries[i], boundaries[i+1])
                      for i in range(len(boundaries)-1)]

    return sorted(breaks)


def analyze():
    """Load corpus CSV and produce all analysis outputs."""
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"No corpus file found at {RAW_CSV}.\n"
            "Run with MODE='collect' first."
        )

    df = pd.read_csv(RAW_CSV, parse_dates=["pub_date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "compound"])
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    print(f"\n=== Corpus summary ===")
    print(f"  Total articles:  {len(df):,}")
    print(f"  Year range:      {df['year'].min()} – {df['year'].max()}")
    print(f"  Median sentiment: {df['compound'].median():.3f}")
    print(f"  Mean sentiment:   {df['compound'].mean():.3f}")
    print(f"\n  Articles per decade:")
    decade_start = (MIN_YEAR // 10) * 10
    decade_end   = (MAX_YEAR // 10) * 10
    decades_all  = list(range(decade_start, decade_end + 10, 10))
    for d in decades_all:
        n = len(df[(df["year"] >= d) & (df["year"] < d+10)])
        print(f"    {d}s: {n:,}")

    # ── Annual series (word-count weighted mean) ─────────────────────────────
    def wavg(g):
        wc = g["word_count"].clip(lower=1)
        return np.average(g["compound"], weights=wc)

    annual = (df.groupby("year")
                .apply(wavg)
                .rename("compound_wavg")
                .reset_index())
    annual.columns = ["year", "compound_wavg"]
    annual = annual.sort_values("year")

    # 3-year rolling mean
    annual["smooth"] = (annual["compound_wavg"]
                        .rolling(3, center=True, min_periods=1)
                        .mean())

    # Article counts per year
    counts = df.groupby("year").size().rename("n")
    annual = annual.merge(counts, on="year")

    # ── Structural break detection ────────────────────────────────────────────
    # Scale break budget with window length; paper predicted ~2 for 1930-1970
    window_years = MAX_YEAR - MIN_YEAR + 1
    max_breaks   = max(3, int(window_years / 18))  # ~7 for 1900-2026
    years_arr    = annual["year"].values
    smooth_arr   = annual["smooth"].values
    break_idxs   = bai_perron_breaks(smooth_arr,
                                     max_breaks=max_breaks, min_segment=5)
    break_years  = [int(years_arr[i]) for i in break_idxs]
    print(f"\n=== Structural breaks (Bai-Perron) ===")
    print(f"  Max breaks allowed: {max_breaks}")
    print(f"  Detected break years: {break_years}")
    print(f"  Paper predicts breaks near 1941 and 1959 (within 1930–1970)")

    # ── Frame shares by decade ────────────────────────────────────────────────
    frame_decade = (df.assign(decade=(df["year"].astype(int)//10)*10)
                      .groupby(["decade","frame"])
                      .size()
                      .unstack(fill_value=0))
    frame_pct = frame_decade.div(frame_decade.sum(axis=1), axis=0) * 100

    # ── FIGURE 1: Sentiment time series ──────────────────────────────────────
    # Wider figure for longer time series
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9),
                                     gridspec_kw={"height_ratios":[3,1],
                                                  "hspace":0.10})
    fig1.patch.set_facecolor("white")

    # Top: compound sentiment
    ax1.bar(annual["year"], annual["compound_wavg"],
            color=[BLUE if v >= 0 else RED
                   for v in annual["compound_wavg"]],
            alpha=0.35, width=0.8, label="_nolegend_")
    ax1.plot(annual["year"], annual["smooth"], "-",
             color=NAVY, lw=2.2, label="3-yr rolling mean")
    ax1.axhline(0, color=GREY, lw=0.8)

    # Mark structural breaks
    y_top = ax1.get_ylim()[1]
    y_bot = ax1.get_ylim()[0]
    for by in break_years:
        ax1.axvline(by, color=RED, lw=1.4, ls="--", alpha=0.8)
        ax1.text(by+0.3, y_top*0.92,
                 f"Break\n{by}", fontsize=8, color=RED,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec=RED, alpha=0.9, lw=0.5))

    # Mark key NYCHA / policy events
    events = [
        (1935, "First\nHouses"),
        (1937, "Wagner-\nSteagall"),
        (1940, "Vladeck"),
        (1949, "Housing\nAct"),
        (1959, "Baruch"),
        (1968, "Fair\nHousing Act"),
        (1974, "Section 8"),
        (1998, "QHWRA"),
        (2013, "NextGen\nNYCHA"),
    ]
    for yr, name in events:
        if MIN_YEAR <= yr <= MAX_YEAR:
            ax1.axvline(yr, color=TEAL, lw=0.8, ls=":", alpha=0.55)
            ax1.text(yr+0.2, y_top*0.72, name,
                     fontsize=6.5, color=TEAL, rotation=90, va="top")

    period_shade(ax1)
    # Only draw period labels if their midpoints are in the visible range
    for x, lbl, col in [(1935.5,"I",PERIOD_COLORS["I"]),
                         (1950,"II",PERIOD_COLORS["II"]),
                         (1964.5,"III",PERIOD_COLORS["III"])]:
        if MIN_YEAR <= x <= MAX_YEAR:
            ax1.text(x, y_bot*0.85, lbl, ha="center",
                     fontsize=9, color=col, style="italic", fontweight="bold")

    ax1.set_ylabel("VADER compound sentiment\n(word-count weighted mean)",
                   fontsize=9.5)
    ax1.set_xlim(MIN_YEAR - 1, MAX_YEAR + 1)
    ax1.legend(fontsize=9, loc="upper right",
               framealpha=0.92, edgecolor=LGREY)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title(f"NYT Sentiment on Public Housing / LES, {MIN_YEAR}–{MAX_YEAR}\n"
                  "Domain-adapted VADER  ·  3-year rolling mean  ·  "
                  "Bai-Perron structural breaks  ·  "
                  f"Paper window ({PAPER_START}–{PAPER_END}) shaded",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.tick_params(labelbottom=False)

    # Bottom: article count
    ax2.bar(annual["year"], annual["n"], color=GREY,
            alpha=0.5, width=0.8)
    ax2.set_ylabel("Articles\nper year", fontsize=8.5)
    ax2.set_xlabel("Year", fontsize=9.5)
    ax2.set_xlim(MIN_YEAR - 1, MAX_YEAR + 1)
    period_shade(ax2, alpha=0.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig1.savefig(os.path.join(OUT_DIR, "fig_sentiment_timeseries.png"),
                 dpi=300, bbox_inches="tight", facecolor="white")
    print("  Saved: fig_sentiment_timeseries.png")
    plt.close(fig1)

    # ── FIGURE 2: Frame shares by decade ────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(max(10, len(decades_all)*0.9), 6))
    fig2.patch.set_facecolor("white")

    frame_order  = ["reform","construction","conflict","pathology","crisis"]
    bottoms = np.zeros(len(decades_all))
    x = np.arange(len(decades_all))

    for frame in frame_order:
        if frame not in frame_pct.columns:
            frame_pct[frame] = 0.0
        vals = [frame_pct.loc[d, frame]
                if d in frame_pct.index else 0.0
                for d in decades_all]
        ax.bar(x, vals, bottom=bottoms,
               color=FRAME_COLORS[frame],
               label=frame.capitalize(),
               edgecolor="white", lw=0.6, width=0.7)
        # Label bars > 8%
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 8:
                ax.text(xi, b + v/2, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in decades_all], fontsize=9,
                       rotation=0 if len(decades_all) <= 8 else 30)
    ax.set_ylabel("Share of articles (%)", fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.92, edgecolor=LGREY, ncol=5,
              bbox_to_anchor=(1.0, 1.08))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"NYT Framing of Public Housing by Decade, {decade_start}s–{decade_end}s\n"
                 "Hypothesized: Reform dominant → Construction → "
                 "Conflict → Pathology/Crisis",
                 fontsize=11, fontweight="bold", pad=10)

    # Shade paper-window decades subtly
    paper_decade_start = (PAPER_START // 10) * 10
    paper_decade_end   = (PAPER_END // 10) * 10
    for d in decades_all:
        if paper_decade_start <= d <= paper_decade_end:
            xi = decades_all.index(d)
            ax.axvspan(xi - 0.4, xi + 0.4, alpha=0.05, color=NAVY, zorder=-1)

    fig2.savefig(os.path.join(OUT_DIR, "fig_frame_shares.png"),
                 dpi=300, bbox_inches="tight", facecolor="white")
    print("  Saved: fig_frame_shares.png")
    plt.close(fig2)

    # ── FIGURE 3: Sentiment by frame over time ───────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(18, 6))
    fig3.patch.set_facecolor("white")

    frame_annual = (df.groupby(["year","frame"])["compound"]
                      .mean()
                      .unstack(fill_value=np.nan))

    for frame in ["reform","pathology","crisis","conflict"]:
        if frame not in frame_annual.columns:
            continue
        col  = FRAME_COLORS[frame]
        vals = frame_annual[frame].rolling(3, center=True,
                                           min_periods=1).mean()
        ax3.plot(frame_annual.index, vals, "-",
                 color=col, lw=1.8,
                 label=frame.capitalize(), alpha=0.85)

    period_shade(ax3)
    ax3.axhline(0, color=GREY, lw=0.8)
    ax3.set_xlabel("Year", fontsize=9.5)
    ax3.set_ylabel("Mean VADER compound score by frame", fontsize=9.5)
    ax3.set_xlim(MIN_YEAR - 1, MAX_YEAR + 1)
    ax3.legend(fontsize=9, loc="lower left",
               framealpha=0.92, edgecolor=LGREY)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_title(f"Sentiment by Narrative Frame, {MIN_YEAR}–{MAX_YEAR}\n"
                  "Reform frame stays positive; Pathology/Crisis "
                  "frames turn more negative over time",
                  fontsize=11, fontweight="bold", pad=10)

    fig3.savefig(os.path.join(OUT_DIR, "fig_sentiment_by_frame.png"),
                 dpi=300, bbox_inches="tight", facecolor="white")
    print("  Saved: fig_sentiment_by_frame.png")
    plt.close(fig3)

    # ── CSV OUTPUTS ──────────────────────────────────────────────────────────
    annual.to_csv(os.path.join(OUT_DIR, "annual_sentiment.csv"), index=False)
    frame_pct.to_csv(os.path.join(OUT_DIR, "frame_shares_by_decade.csv"))

    # Break comparison table — only compare breaks in the paper's window
    paper_breaks        = [1941, 1959]
    breaks_paper_window = [b for b in break_years
                           if PAPER_START <= b <= PAPER_END]
    n_compare = min(len(breaks_paper_window), len(paper_breaks))
    comparison = pd.DataFrame({
        "break_year":     breaks_paper_window[:n_compare],
        "paper_boundary": paper_breaks[:n_compare],
        "match":          [abs(b-p) <= 3
                           for b,p in zip(breaks_paper_window[:n_compare],
                                          paper_breaks[:n_compare])]
    })
    comparison.to_csv(os.path.join(OUT_DIR, "break_comparison.csv"),
                      index=False)

    # Full break list (including those outside paper window)
    all_breaks_df = pd.DataFrame({
        "break_year":    break_years,
        "in_paper_window": [PAPER_START <= b <= PAPER_END
                            for b in break_years],
    })
    all_breaks_df.to_csv(os.path.join(OUT_DIR, "all_structural_breaks.csv"),
                         index=False)

    # ── PRINT SUMMARY TABLE ──────────────────────────────────────────────────
    print("\n=== Annual sentiment (first 5 rows) ===")
    print(annual.head().to_string(index=False))

    print("\n=== Frame shares by decade (%) ===")
    print(frame_pct.round(1).to_string())

    print("\n=== Structural break comparison (paper window only) ===")
    if len(comparison) > 0:
        print(comparison.to_string(index=False))
    else:
        print("  No breaks detected within 1930–1970.")

    print("\n=== All detected breaks ===")
    print(all_breaks_df.to_string(index=False))

    print(f"\nAll outputs written to: {os.path.abspath(OUT_DIR)}/")
    print("  fig_sentiment_timeseries.png  — main time-series figure")
    print("  fig_frame_shares.png          — frame shares by decade")
    print("  fig_sentiment_by_frame.png    — sentiment within each frame")
    print("  annual_sentiment.csv          — year-level data")
    print("  frame_shares_by_decade.csv    — frame share table")
    print("  break_comparison.csv          — break year vs paper periods")
    print("  all_structural_breaks.csv     — full break list with flag")
    print("  nyt_raw_corpus.csv            — full article corpus")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"NYT Sentiment Analysis  |  MODE = {MODE}")
    print(f"Window: {MIN_YEAR}–{MAX_YEAR}  (paper window: {PAPER_START}–{PAPER_END})")
    print(f"Output directory: {os.path.abspath(OUT_DIR)}/\n")

    if MODE in ("collect", "both"):
        print("=== PHASE 1: Data collection ===")
        collect()

    if MODE in ("analyze", "both"):
        print("\n=== PHASE 2: Analysis ===")
        analyze()