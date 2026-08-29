import pandas as pd
import matplotlib.pyplot as plt

# --- LOAD DATA ---
df = pd.read_csv("sentiment_output/nyt_raw_corpus.csv")


# --- DEFINE YOUR TOPIC FILTER ---
# Adjust this depending on your column names
topic_mask = (
    (df["les"] == 1) | 
    (df["public_housing"] == 1)
)

# --- COUNT ARTICLES ---
# topic articles per year
topic_counts = (
    df[topic_mask]
    .groupby("year")
    .size()
    .rename("topic_articles")
)

# total NYT articles per year
total_counts = (
    df.groupby("year")
    .size()
    .rename("total_articles")
)

# --- MERGE + NORMALIZE ---
counts = pd.concat([topic_counts, total_counts], axis=1).fillna(0)

counts["share"] = counts["topic_articles"] / counts["total_articles"]

# --- SMOOTH (optional, recommended) ---
counts["share_smooth"] = counts["share"].rolling(3, center=True).mean()

# --- PLOT ---
fig, ax = plt.subplots(figsize=(12, 4))

# bars = raw normalized share
ax.bar(counts.index, counts["share"], color="gray", alpha=0.6, label="Annual share")

# line = smoothed
ax.plot(counts.index, counts["share_smooth"], color="black", linewidth=2, label="3-yr rolling mean")

ax.set_ylabel("Share of NYT articles")
ax.set_xlabel("Year")
ax.set_title("NYT Coverage Intensity (Normalized)")

ax.legend()
plt.tight_layout()
plt.show()