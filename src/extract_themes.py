import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("data/processed/sentiment_output.csv")

def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

df["lemmas"] = df["clean_text"].dropna().apply(preprocess)

# TF-IDF extraction
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=150)
X = vectorizer.fit_transform(df["lemmas"])
keywords = vectorizer.get_feature_names_out()

def extract_keywords(text):
    return [word for word in keywords if word in text]

df["keywords"] = df["lemmas"].apply(extract_keywords)

# Manual clustering
theme_map = {
    "Account Access Issues": ["login", "password", "signin", "otp"],
    "Transaction Performance": ["transfer", "delay", "transaction", "slow"],
    "User Interface & UX": ["ui", "screen", "navigation", "design"],
    "Customer Support": ["support", "agent", "help", "response"],
    "App Reliability": ["crash", "freeze", "error", "bug"]
}

def assign_themes(keywords):
    themes = []
    for theme, kws in theme_map.items():
        if any(kw in keywords for kw in kws):
            themes.append(theme)
    return themes if themes else ["Other"]

df["themes"] = df["keywords"].apply(assign_themes)
df.to_csv("data/processed/sentiment_themes.csv", index=False)
print("✅ Themes saved to: data/processed/sentiment_themes.csv")
