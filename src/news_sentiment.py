import requests
import pandas as pd
import time
from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import API keys
from src.config import NEWS_API_KEY
# A VADER sentiment analyzer instance for calculating sentiment scores.
analyzer = SentimentIntensityAnalyzer()

def get_news_sentiment(query: str, ticker: str, start_date: date = None, end_date: date = None) -> pd.DataFrame:
    """
    Fetches news headlines for a specific company and calculates sentiment.
    This function can operate in two modes:
    1.  **Date-Driven (for historical backfill):** If start_date and end_date are provided,
        it fetches all articles for that specific day. This is the best method for
        building a complete time-series dataset.
    2.  **Default (for most popular/recent):** If no dates are provided, it fetches the
        most relevant and popular articles, paginating through them to get as many
        as possible up to the page limit. This is a good way to quickly gather
        impactful news without backfilling.

    Args:
        query (str): The search phrase (e.g., full company name).
        ticker (str): The stock ticker (e.g., 'WMT').
        api_key (str): Your News API key.
        start_date (date, optional): The start date for the search. Defaults to None.
        end_date (date, optional): The end date for the search. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with the date, ticker, and sentiment scores.
    """
    url = 'https://newsapi.org/v2/everything'
    all_articles = []
    page = 1
    total_results = 1 # Initialize to enter the loop

    while len(all_articles) < total_results:
        # Construct parameters. The 'from' and 'to' are included only if dates are provided.
        params = {
            'q': query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'relevancy',
            'page': page
        }
        
        # Add date parameters if they were provided to the function call.
        if start_date and end_date:
            params['from'] = start_date.isoformat()
            params['to'] = end_date.isoformat()

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

        if data['status'] != 'ok':
            print(f"API request failed for {ticker}: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
        
        articles = data.get('articles', [])
        if not articles:
            print(f"No more articles found for {query}")
            break

        all_articles.extend(articles)
        total_results = data.get('totalResults', 0)
        
        if start_date and end_date:
            if len(all_articles) >= total_results:
                break
        else:
            break

        page += 1
        #time.sleep(1)

    sentiment_list = []
    for article in all_articles:
        text_to_analyze = f"{article.get('title', '')} {article.get('description', '')}"
        vs = analyzer.polarity_scores(text_to_analyze)
        
        sentiment_list.append({
            'Date': start_date if start_date else date.today(),
            'Ticker': ticker,
            'Positive_Sentiment': vs['pos'],
            'Negative_Sentiment': vs['neg'],
            'Neutral_Sentiment': vs['neu'],
            'Compound_Sentiment': vs['compound'],
        })

    return pd.DataFrame(sentiment_list)

