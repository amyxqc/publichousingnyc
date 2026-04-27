"""
NYT Sentiment Analysis — Chen (2026) LES Public Housing Paper
=============================================================
EXTENDED VERSION: 1900 → present (2026)
with QUOTA-AWARE RESUMPTION (v2)

Changes vs. v1:
  • Tracks completed (year, month) pairs in a state file so resume
    skips months you've already hit — no wasted quota re-requesting them.
  • Detects the NYT daily 500-request limit: if a 429 persists after the
    60-second per-minute sleep, the script exits gracefully instead of
    looping through error logs.
  • Prints how many requests you've used this session so you can see
    the daily budget draining in real time.

Run the same way as before. If you hit the daily cap, wait ~24 hours
and re-run with MODE = "collect" — it will pick up exactly where it
left off.
"""

# ─── USER SETTINGS ────────────────────────────────────────────────────────────
NYT_API_KEY = "7DdBw359pt8APA0PFXSdRNdoZz1d3eo2ek6AAOI0ak7eYRFz"
MODE        = "both"
OUT_DIR     = "./sentiment_output"

MIN_YEAR    = 1900
MAX_YEAR    = 2026

PAPER_START = 1930
PAPER_END   = 1970

# NYT Archive API daily limit. If we hit 2 consecutive 429s after the
# per-minute sleep, we assume we've exhausted this and stop.
DAILY_REQUEST_BUDGET = 500
# ──────────────────────────────────────────────────────────────────────────────

import os, time, json, re, csv, math, sys
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
# 1. DOMAIN-ADAPTED VADER LEXICON  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

POSITIVE_TERMS = {
    "modern": 2.0, "sanitary": 2.5, "decent": 2.0, "spacious": 2.0,
    "ventilated": 1.5, "improved": 1.5, "clearance": 1.0, "wholesome": 2.0,
    "adequate": 1.5, "rehabilitation": 1.5, "revitalization": 1.5, "relief": 1.5,
}

NEGATIVE_TERMS = {
    "slum": -2.5, "slums": -2.5, "blight": -2.5, "blighted": -2.5,
    "decrepit": -2.0, "vermin": -2.5, "overcrowded": -2.0, "overcrowding": -2.0,
    "dilapidated": -2.0, "squalid": -3.0, "squalor": -3.0, "crime-ridden": -2.5,
    "delinquency": -1.5, "delinquent": -1.5, "gang": -2.0, "gangs": -2.0,
    "vandalism": -2.0, "deterioration": -2.0, "deteriorating": -2.0,
    "abandoned": -2.0, "abandonment": -2.0, "neglect": -2.0, "neglected": -2.0,
    "rundown": -2.0,
}

DRIFT_TERMS = {
    "project":   {"before": 0.0,  "after": -1.0, "cutoff": 1950},
    "projects":  {"before": 0.0,  "after": -1.0, "cutoff": 1950},
    "clearance": {"before": 1.0,  "after": -1.5, "cutoff": 1955},
    "welfare":   {"before": 0.5,  "after": -1.0, "cutoff": 1955},
    "troubled":  {"before": -1.0, "after": -1.5, "cutoff": 1960},
}


def build_vader(year):
    analyzer = SentimentIntensityAnalyzer()
    for term, score in POSITIVE_TERMS.items():
        analyzer.lexicon[term] = score
    for term, score in NEGATIVE_TERMS.items():
        analyzer.lexicon[term] = score
    for term, cfg in DRIFT_TERMS.items():
        score = cfg["before"] if year < cfg["cutoff"] else cfg["after"]
        analyzer.lexicon[term] = score
    return analyzer


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KEYWORD FILTERS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

TIER1_TERMS = [
    "public housing", "housing project", "housing projects", "nycha",
    "housing authority", "slum clearance", "urban renewal", "tenement",
    "tenements", "first houses", "vladeck houses", "laguardia houses",
    "la guardia houses", "baruch houses", "alfred e. smith houses",
    "lillian wald houses", "jacob riis houses", "wald houses", "riis houses",
    "smith houses", "gompers houses", "rutgers houses",
]

TIER2_GEO = ["lower east side", "east side", "manhattan",
             "new york city", "new york"]

BROAD_TERMS = {"tenement", "tenements", "housing authority"}

EXCLUDE_SECTION = {"real estate"}
EXCLUDE_MATERIAL = {"letter", "letters", "correction", "corrections",
                    "paid notice", "classified", "obituaries"}


def article_text(article):
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
    text = article_text(article)
    material = (article.get("type_of_material") or "").lower()
    section  = (article.get("section_name") or "").lower()

    if any(exc in material for exc in EXCLUDE_MATERIAL):
        return False

    matched_tier1 = [t for t in TIER1_TERMS if t in text]
    if not matched_tier1:
        return False

    non_broad = [t for t in matched_tier1 if t not in BROAD_TERMS]
    if not non_broad:
        if not any(g in text for g in TIER2_GEO):
            return False

    if section in EXCLUDE_SECTION:
        if "lower east side" not in text and "east side" not in text:
            return False

    if "public housing" in matched_tier1 and year < 1933:
        other = [t for t in matched_tier1 if t != "public housing"]
        if not other:
            return False

    if "urban renewal" in matched_tier1 and year < 1949:
        return False

    if "nycha" in matched_tier1 and year < 1934:
        other = [t for t in matched_tier1 if t != "nycha"]
        if not other:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FRAMING CATEGORIES  (unchanged)
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
    scores = {}
    for frame, markers in FRAMES.items():
        scores[frame] = sum(1 for m in markers if m in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "construction"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCORING FUNCTION  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def score_article(article, year):
    text  = article_text(article)
    analyzer = build_vader(year)
    vs    = analyzer.polarity_scores(text)

    words = text.split()
    n     = max(len(words), 1)
    reform_idx   = sum(1 for w in words if w in POSITIVE_TERMS) / n
    pathology_idx= sum(1 for w in words if w in NEGATIVE_TERMS) / n
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
# 5. DATA COLLECTION — with state file + daily-limit detection
# ═══════════════════════════════════════════════════════════════════════════════

RAW_CSV    = os.path.join(OUT_DIR, "nyt_raw_corpus.csv")
STATE_JSON = os.path.join(OUT_DIR, "collection_state.json")

FIELDNAMES = [
    "id", "pub_date", "year", "headline", "section", "material",
    "word_count", "compound", "pos", "neg", "neu",
    "reform_idx", "pathology_idx", "frame_ratio", "frame"
]


def months_for_year(year, now):
    if year < now.year:
        return 12
    if year == now.year:
        return max(0, now.month - 1)
    return 0


def load_state():
    """Return set of 'YYYY-MM' strings for months already fetched."""
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON) as f:
            data = json.load(f)
            return set(data.get("completed_months", []))
    return set()


def save_state(completed):
    with open(STATE_JSON, "w") as f:
        json.dump({"completed_months": sorted(completed),
                   "last_updated": datetime.now().isoformat()}, f, indent=2)


def bootstrap_state_from_csv():
    """
    If a CSV exists from an older run but no state file, assume every
    month between the earliest and latest article pub_date was completed.
    This is conservative — if a month in that range genuinely had zero
    matches, we'll skip it on resume (acceptable loss).
    """
    if not os.path.exists(RAW_CSV):
        return set()
    df = pd.read_csv(RAW_CSV, parse_dates=["pub_date"], low_memory=False)
    if df.empty:
        return set()
    mn, mx = df["pub_date"].min(), df["pub_date"].max()
    completed = set()
    y, m = mn.year, mn.month
    while (y, m) <= (mx.year, mx.month):
        completed.add(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    print(f"  Bootstrapped state from CSV: assuming {len(completed)} "
          f"months ({mn.date()} to {mx.date()}) already collected.")
    print(f"  If that's wrong, delete {STATE_JSON} and the script "
          f"will re-check every month.")
    return completed


def collect():
    if NYT_API_KEY == "YOUR_KEY_HERE":
        raise ValueError("Set NYT_API_KEY at the top of this script.")

    now = datetime.now()

    # Load state — try JSON first, then bootstrap from CSV if needed
    completed = load_state()
    if not completed and os.path.exists(RAW_CSV):
        completed = bootstrap_state_from_csv()
        save_state(completed)

    # Build todo list
    todo = []
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        for month in range(1, months_for_year(year, now) + 1):
            key = f"{year:04d}-{month:02d}"
            if key not in completed:
                todo.append((year, month, key))

    print(f"  Range: {MIN_YEAR}–{MAX_YEAR}")
    print(f"  Months already done: {len(completed)}")
    print(f"  Months remaining:    {len(todo)}")

    if not todo:
        print("  Nothing to collect — all months done.")
        return

    # Cap today's run at the daily budget (leave 5-request buffer)
    runnable = min(len(todo), DAILY_REQUEST_BUDGET - 5)
    if runnable < len(todo):
        days_needed = math.ceil(len(todo) / DAILY_REQUEST_BUDGET)
        print(f"  Daily budget: {DAILY_REQUEST_BUDGET} requests.")
        print(f"  Will attempt {runnable} months today. "
              f"Need ~{days_needed} days total.\n")
    else:
        print(f"  Will finish in one run (~{runnable * 12 / 60:.0f} min).\n")

    # Load existing rows
    seen_ids = set()
    rows     = []
    if os.path.exists(RAW_CSV):
        existing = pd.read_csv(RAW_CSV, low_memory=False)
        seen_ids = set(existing["id"].astype(str).tolist())
        rows     = existing.to_dict("records")
        print(f"  Loaded {len(rows)} existing articles.\n")

    consecutive_429s  = 0
    requests_this_run = 0

    for year, month, key in todo[:runnable]:
        url = (f"https://api.nytimes.com/svc/archive/v1/"
               f"{year}/{month}.json?api-key={NYT_API_KEY}")

        try:
            resp = requests.get(url, timeout=30)
            requests_this_run += 1

            if resp.status_code == 429:
                print(f"  429 at {key} — sleeping 60s (per-minute throttle?)…")
                time.sleep(60)
                resp = requests.get(url, timeout=30)
                requests_this_run += 1

                if resp.status_code == 429:
                    consecutive_429s += 1
                    print(f"  Still 429 after sleep. "
                          f"Consecutive: {consecutive_429s}")
                    if consecutive_429s >= 2:
                        print("\n" + "="*70)
                        print("  DAILY RATE LIMIT LIKELY EXHAUSTED")
                        print(f"  (NYT caps at {DAILY_REQUEST_BUDGET}/day)")
                        print(f"  Requests this run: {requests_this_run}")
                        print("  Stopping cleanly. Re-run tomorrow.")
                        print("="*70)
                        break
                    continue
                else:
                    consecutive_429s = 0

            resp.raise_for_status()
            data = resp.json()
            articles = data.get("response", {}).get("docs", [])
            consecutive_429s = 0

        except Exception as e:
            print(f"  ERROR {key}: {e}")
            # Do NOT mark completed — we'll retry next run
            time.sleep(12)
            continue

        month_count = 0
        for art in articles:
            aid = str(art.get("_id", ""))
            if aid in seen_ids:
                continue
            if passes_keyword_filter(art, year):
                row = score_article(art, year)
                rows.append(row)
                seen_ids.add(aid)
                month_count += 1

        completed.add(key)
        print(f"  {key}: {len(articles):5d} total, {month_count:3d} matched  "
              f"(corpus: {len(rows)}, run reqs: {requests_this_run})")

        pd.DataFrame(rows, columns=FIELDNAMES).to_csv(RAW_CSV, index=False)
        save_state(completed)

        time.sleep(12)

    print(f"\nRun complete. Corpus: {len(rows)} articles. "
          f"Completed: {len(completed)} months.")

    remaining = [k for y, m, k in todo if k not in completed]
    if remaining:
        print(f"  Still to collect: {len(remaining)} months "
              f"(from {remaining[0]} to {remaining[-1]}).")
        print("  Re-run tomorrow to continue.")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANALYSIS  (same as v1)
# ═══════════════════════════════════════════════════════════════════════════════

NAVY='#1a3a5c'; BLUE='#2166ac'; MID='#4393c3'; RED='#c0392b'
AMBER='#b47a14'; TEAL='#006464'; GREY='#666666'; LGREY='#cccccc'

FRAME_COLORS = {
    "reform": '#2166ac', "construction": '#aaaaaa',
    "conflict": '#e07b39', "pathology": '#c0392b', "crisis": '#7b1a1a',
}
PERIOD_COLORS = {'I': '#1a3a5c', 'II': '#2166ac', 'III': '#4393c3'}


def period_shade(ax, alpha=0.07):
    ax.axvspan(1930, 1941, alpha=alpha, color=PERIOD_COLORS['I'],  zorder=0)
    ax.axvspan(1942, 1959, alpha=alpha, color=PERIOD_COLORS['II'], zorder=0)
    ax.axvspan(1960, 1970, alpha=alpha, color=PERIOD_COLORS['III'],zorder=0)
    for xv in [1941, 1959]:
        ax.axvline(xv, color=GREY, lw=0.9, ls='--', alpha=0.5)
    for xv in [PAPER_START, PAPER_END]:
        ax.axvline(xv, color=GREY, lw=0.6, ls=':', alpha=0.4)


def bai_perron_breaks(series, max_breaks=3, min_segment=5):
    n = len(series); y = np.array(series)
    def ssr(seg): return np.sum((seg - seg.mean())**2) if len(seg) > 0 else 0.0
    def find_one_break(y_sub, min_seg):
        best_ssr = np.inf; best_bp = None
        for bp in range(min_seg, len(y_sub) - min_seg + 1):
            s = ssr(y_sub[:bp]) + ssr(y_sub[bp:])
            if s < best_ssr:
                best_ssr = s; best_bp = bp
        return best_bp, best_ssr
    breaks = []; segments = [(0, n)]
    for _ in range(max_breaks):
        best_gain = 0; best_new_bp = None
        for si, (start, end) in enumerate(segments):
            seg = y[start:end]
            if (end - start) < 2 * min_segment: continue
            bp, new_ssr = find_one_break(seg, min_segment)
            if bp is None: continue
            old_ssr = ssr(seg); gain = old_ssr - new_ssr
            if gain > best_gain:
                best_gain = gain; best_new_bp = start + bp
        if best_new_bp is None or best_gain < 1e-6: break
        breaks.append(best_new_bp); breaks.sort()
        boundaries = [0] + breaks + [n]
        segments = [(boundaries[i], boundaries[i+1])
                    for i in range(len(boundaries)-1)]
    return sorted(breaks)


def analyze():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"No corpus at {RAW_CSV}. Collect first.")

    df = pd.read_csv(RAW_CSV, parse_dates=["pub_date"], low_memory=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "compound"])
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    print(f"\n=== Corpus summary ===")
    print(f"  Total articles:   {len(df):,}")
    print(f"  Year range:       {df['year'].min()} – {df['year'].max()}")
    print(f"  Median sentiment: {df['compound'].median():.3f}")
    print(f"  Mean sentiment:   {df['compound'].mean():.3f}")

    decade_start = (int(df['year'].min()) // 10) * 10
    decade_end   = (int(df['year'].max()) // 10) * 10
    decades_all  = list(range(decade_start, decade_end + 10, 10))
    print(f"\n  Articles per decade:")
    for d in decades_all:
        n = len(df[(df["year"] >= d) & (df["year"] < d+10)])
        print(f"    {d}s: {n:,}")

    def wavg(g):
        wc = g["word_count"].clip(lower=1)
        return np.average(g["compound"], weights=wc)

    annual = (df.groupby("year").apply(wavg)
                .rename("compound_wavg").reset_index())
    annual.columns = ["year", "compound_wavg"]
    annual = annual.sort_values("year")
    annual["smooth"] = annual["compound_wavg"].rolling(
        3, center=True, min_periods=1).mean()
    counts = df.groupby("year").size().rename("n")
    annual = annual.merge(counts, on="year")

    window_years = int(annual["year"].max() - annual["year"].min() + 1)
    max_breaks   = max(3, window_years // 18)
    break_idxs   = bai_perron_breaks(annual["smooth"].values,
                                     max_breaks=max_breaks, min_segment=5)
    break_years  = [int(annual["year"].values[i]) for i in break_idxs]
    print(f"\n=== Structural breaks (Bai-Perron) ===")
    print(f"  Max breaks allowed: {max_breaks}")
    print(f"  Detected: {break_years}")

    frame_decade = (df.assign(decade=(df["year"].astype(int)//10)*10)
                      .groupby(["decade","frame"]).size()
                      .unstack(fill_value=0))
    frame_pct = frame_decade.div(frame_decade.sum(axis=1), axis=0) * 100

    # FIGURE 1
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9),
                                     gridspec_kw={"height_ratios":[3,1],
                                                  "hspace":0.10})
    fig1.patch.set_facecolor("white")
    ax1.bar(annual["year"], annual["compound_wavg"],
            color=[BLUE if v >= 0 else RED for v in annual["compound_wavg"]],
            alpha=0.35, width=0.8)
    ax1.plot(annual["year"], annual["smooth"], "-",
             color=NAVY, lw=2.2, label="3-yr rolling mean")
    ax1.axhline(0, color=GREY, lw=0.8)
    y_top = ax1.get_ylim()[1]
    for by in break_years:
        ax1.axvline(by, color=RED, lw=1.4, ls="--", alpha=0.8)
        ax1.text(by+0.3, y_top*0.92, f"Break\n{by}", fontsize=8, color=RED,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec=RED, alpha=0.9, lw=0.5))

    events = [(1935,"First\nHouses"),(1937,"Wagner-\nSteagall"),
              (1940,"Vladeck"),(1949,"Housing\nAct"),(1959,"Baruch"),
              (1968,"Fair\nHousing Act"),(1974,"Section 8"),
              (1998,"QHWRA"),(2013,"NextGen\nNYCHA")]
    for yr, name in events:
        if MIN_YEAR <= yr <= MAX_YEAR:
            ax1.axvline(yr, color=TEAL, lw=0.8, ls=":", alpha=0.55)
            ax1.text(yr+0.2, y_top*0.72, name, fontsize=6.5,
                     color=TEAL, rotation=90, va="top")

    period_shade(ax1)
    ax1.set_ylabel("VADER compound sentiment\n(word-count weighted mean)",
                   fontsize=9.5)
    ax1.set_xlim(MIN_YEAR - 1, MAX_YEAR + 1)
    ax1.legend(fontsize=9, loc="upper right",
               framealpha=0.92, edgecolor=LGREY)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title(f"NYT Sentiment on Public Housing / LES, "
                  f"{MIN_YEAR}–{MAX_YEAR}",                                                                                                            
                  fontsize=11, fontweight="bold", pad=10)
    ax1.tick_params(labelbottom=False)

    ax2.bar(annual["year"], annual["n"], color=GREY, alpha=0.5, width=0.8)
    ax2.set_ylabel("Articles\nper year", fontsize=8.5)
    ax2.set_xlabel("Year", fontsize=9.5)
    ax2.set_xlim(MIN_YEAR - 1, MAX_YEAR + 1)
    period_shade(ax2, alpha=0.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig1.savefig(os.path.join(OUT_DIR, "fig_sentiment_timeseries.png"),
                 dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # FIGURE 2
    fig2, ax = plt.subplots(figsize=(max(10, len(decades_all)*0.9), 6))
    fig2.patch.set_facecolor("white")
    frame_order = ["reform","construction","conflict","pathology","crisis"]
    bottoms = np.zeros(len(decades_all)); x = np.arange(len(decades_all))
    for frame in frame_order:
        if frame not in frame_pct.columns:
            frame_pct[frame] = 0.0
        vals = [frame_pct.loc[d, frame] if d in frame_pct.index else 0.0
                for d in decades_all]
        ax.bar(x, vals, bottom=bottoms, color=FRAME_COLORS[frame],
               label=frame.capitalize(), edgecolor="white", lw=0.6, width=0.7)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 8:
                ax.text(xi, b + v/2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in decades_all], fontsize=9)
    ax.set_ylabel("Share of articles (%)", fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92,
              edgecolor=LGREY, ncol=5, bbox_to_anchor=(1.0, 1.08))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(f"NYT Framing by Decade, {decade_start}s–{decade_end}s",
                 fontsize=11, fontweight="bold", pad=10)
    fig2.savefig(os.path.join(OUT_DIR, "fig_frame_shares.png"),
                 dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    # FIGURE 3
    fig3, ax3 = plt.subplots(figsize=(18, 6))
    fig3.patch.set_facecolor("white")
    frame_annual = (df.groupby(["year","frame"])["compound"].mean()
                      .unstack(fill_value=np.nan))
    for frame in ["reform","pathology","crisis","conflict"]:
        if frame not in frame_annual.columns: continue
        vals = frame_annual[frame].rolling(3, center=True, min_periods=1).mean()
        ax3.plot(frame_annual.index, vals, "-",
                 color=FRAME_COLORS[frame], lw=1.8,
                 label=frame.capitalize(), alpha=0.85)
    period_shade(ax3)
    ax3.axhline(0, color=GREY, lw=0.8)
    ax3.set_xlabel("Year", fontsize=9.5)
    ax3.set_ylabel("Mean VADER compound by frame", fontsize=9.5)
    ax3.set_xlim(MIN_YEAR - 1, MAX_YEAR + 1)
    ax3.legend(fontsize=9, loc="lower left",
               framealpha=0.92, edgecolor=LGREY)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.set_title(f"Sentiment by Frame, {MIN_YEAR}–{MAX_YEAR}",
                  fontsize=11, fontweight="bold", pad=10)
    fig3.savefig(os.path.join(OUT_DIR, "fig_sentiment_by_frame.png"),
                 dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig3)

    annual.to_csv(os.path.join(OUT_DIR, "annual_sentiment.csv"), index=False)
    frame_pct.to_csv(os.path.join(OUT_DIR, "frame_shares_by_decade.csv"))

    paper_breaks = [1941, 1959]
    breaks_paper_window = [b for b in break_years
                           if PAPER_START <= b <= PAPER_END]
    n_compare = min(len(breaks_paper_window), len(paper_breaks))
    comparison = pd.DataFrame({
        "break_year":     breaks_paper_window[:n_compare],
        "paper_boundary": paper_breaks[:n_compare],
        "match": [abs(b-p) <= 3
                  for b,p in zip(breaks_paper_window[:n_compare],
                                 paper_breaks[:n_compare])]
    })
    comparison.to_csv(os.path.join(OUT_DIR, "break_comparison.csv"), index=False)

    pd.DataFrame({
        "break_year": break_years,
        "in_paper_window": [PAPER_START <= b <= PAPER_END
                            for b in break_years]
    }).to_csv(os.path.join(OUT_DIR, "all_structural_breaks.csv"), index=False)

    print(f"\nAll outputs in: {os.path.abspath(OUT_DIR)}/")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"NYT Sentiment Analysis (v2)  |  MODE = {MODE}")
    print(f"Window: {MIN_YEAR}–{MAX_YEAR}  "
          f"(paper: {PAPER_START}–{PAPER_END})")
    print(f"Output: {os.path.abspath(OUT_DIR)}/\n")

    if MODE in ("collect", "both"):
        print("=== PHASE 1: Data collection ===")
        collect()

    if MODE in ("analyze", "both"):
        print("\n=== PHASE 2: Analysis ===")
        analyze()