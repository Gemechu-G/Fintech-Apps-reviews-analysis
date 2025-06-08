import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load data
df = pd.read_csv("data/processed/sentiment_themes.csv")

# Ensure output folder exists
import os
os.makedirs("reports/figures", exist_ok=True)

# -------------------------
# 1. Sentiment by Bank
# -------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="bank", hue="sentiment", palette="Set2")
plt.title("Sentiment Distribution by Bank")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("reports/figures/sentiment_by_bank.png")
plt.close()

# -------------------------
# 2. Rating Distribution by Sentiment
# -------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="sentiment", y="rating", palette="coolwarm")
plt.title("Rating Distribution by Sentiment")
plt.tight_layout()
plt.savefig("reports/figures/rating_by_sentiment.png")
plt.close()

# -------------------------
# 3. Word Cloud per Bank (All Reviews)
# -------------------------
for bank in df['bank'].unique():
    bank_text = " ".join(df[df['bank'] == bank]['clean_text'].dropna().tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(bank_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {bank}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/wordcloud_{bank.replace(' ', '_')}.png")
    plt.close()

# -------------------------
# 4. Top Keywords (Global)
# -------------------------
from collections import Counter

all_keywords = sum(df['keywords'].dropna().tolist(), [])
keyword_counts = Counter(all_keywords)
top_keywords = keyword_counts.most_common(10)

keys, vals = zip(*top_keywords)
plt.figure(figsize=(8, 5))
sns.barplot(x=vals, y=keys, palette="viridis")
plt.title("Top 10 Keywords from Reviews")
plt.tight_layout()
plt.savefig("reports/figures/top_keywords.png")
plt.close()

print("✅ Visualizations saved to `reports/figures/`")
