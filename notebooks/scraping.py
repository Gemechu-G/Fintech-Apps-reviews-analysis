from google_play_scraper import Sort, reviews
import pandas as pd
from datetime import datetime

def scrape_reviews(app_id, bank_name, max_reviews=400):
    all_reviews = []
    next_token = None

    while len(all_reviews) < max_reviews:
        result, next_token = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=100,
            continuation_token=next_token
        )
        for r in result:
            all_reviews.append({
                'review': r['content'],
                'rating': r['score'],
                'date': r['at'].strftime('%Y-%m-%d'),
                'bank': bank_name,
                'source': 'Google Play'
            })
        if not next_token:
            break

    return pd.DataFrame(all_reviews[:max_reviews])

# App IDs (adjust if needed)
apps = {
    'CBE': 'com.combankethio.mobilebanking',
    'BOA': 'com.abyssiniabank.mobilebanking',
    'Dashen': 'com.dashenbank.app'
}

all_data = []

for bank, app_id in apps.items():
    print(f"Scraping {bank}...")
    df = scrape_reviews(app_id, bank)
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

# Clean: remove duplicates, missing
final_df = final_df.dropna(subset=['review'])
final_df = final_df.drop_duplicates()

# Save to file
final_df.to_csv('data/processed/cleaned_bank_reviews.csv', index=False)
print("✅ Done: Saved to cleaned_bank_reviews.csv")
