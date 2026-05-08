import pandas as pd
import requests
import os
import time
from tqdm import tqdm


TMDB_API_KEY = "2d771e7902402ece8d8e483f6e1d639c"
MOVIES_DAT_PATH = "../data/ml-1m/movies.dat" 
OUTPUT_PATH = "../data/processed/movies_enriched.csv"

def fetch_tmdb_data(title, year):
    
    search_url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "primary_release_year": year,
        "language": "en-US"
    }
    
    try:
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                movie = results[0]
                return {
                    "overview": movie.get("overview", ""),
                    "poster_path": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else "",
                    "tmdb_id": movie.get("id", "")
                }
    except Exception as e:
        print(f"Error fetching {title}: {e}")
    
    return {"overview": "", "poster_path": "", "tmdb_id": ""}

def main():
    print("Processing ...")
    
    movies = pd.read_csv(MOVIES_DAT_PATH, sep='::', header=None, 
                         names=['MovieID', 'Title', 'Genres'], engine='python', encoding='latin-1')
    
    
    movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)')
    movies['Clean_Title'] = movies['Title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

    print("...")
    
    # Small-scale test
    # test_movies = movies.head(100).copy() 
    
    test_movies = movies.copy() 
    
    overviews = []
    poster_paths = []
    tmdb_ids = []

    
    for _, row in tqdm(test_movies.iterrows(), total=len(test_movies)):
        data = fetch_tmdb_data(row['Clean_Title'], row['Year'])
        overviews.append(data['overview'])
        poster_paths.append(data['poster_path'])
        tmdb_ids.append(data['tmdb_id'])
        time.sleep(0.2) # Rate limiter

    test_movies['Overview'] = overviews
    test_movies['Poster_URL'] = poster_paths
    test_movies['TMDB_ID'] = tmdb_ids

    
    os.makedirs("../data/processed/", exist_ok=True)
    test_movies.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Done. File is saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
