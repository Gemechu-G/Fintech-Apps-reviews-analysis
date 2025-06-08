import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('vader_lexicon')

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_bank_reviews.csv")

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['clean_text'].apply(classify_sentiment)

# Keyword Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
X = tfidf.fit_transform(df['clean_text'])
keywords = tfidf.get_feature_names_out()

def extract_keywords(text):
    return [kw for kw in keywords if kw in text]

df['keywords'] = df['clean_text'].apply(extract_keywords)

# Save
df.to_csv("data/processed/sentiment_themes.csv", index=False)
print("✅ Sentiment and themes saved.")
