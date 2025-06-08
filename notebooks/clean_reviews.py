import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Load language model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("ethiopian_bank_reviews.csv")

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ""

    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = re.sub(r'\d+', '', text)                  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)              # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()         # Remove extra whitespace

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(lemmatized)

# Apply cleaning
df['clean_text'] = df['review_text'].apply(clean_text)

# Drop empty/duplicate
df = df[df['clean_text'].str.strip() != '']
df = df.drop_duplicates(subset=['clean_text'])

# Save cleaned version
df.to_csv("cleaned_bank_reviews.csv", index=False)
print("✅ Text cleaning complete. Cleaned file saved.")
