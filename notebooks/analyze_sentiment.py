import pandas as pd
from transformers import pipeline
from tqdm import tqdm

tqdm.pandas()
df = pd.read_csv("data/processed/clean_reviews.csv")

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze(text):
    try:
        res = classifier(text[:512])[0]
        label = res["label"]
        score = res["score"]
        return pd.Series([label, score])
    except:
        return pd.Series(["NEUTRAL", 0.5])

df[["sentiment", "sentiment_score"]] = df["clean_text"].progress_apply(analyze)
df.to_csv("data/processed/sentiment_output.csv", index=False)
print("✅ Sentiment analysis saved to: data/processed/sentiment_output.csv")

agg = df.groupby(['bank', 'rating'])['sentiment_score'].mean().reset_index()
print(agg)