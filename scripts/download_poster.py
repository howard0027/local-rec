import pandas as pd
import requests
import os
import concurrent.futures

# Configuration
CSV_PATH = "./data/processed/movies_enriched.csv"
SAVE_DIR = "./data/posters"
MAX_THREADS = 10 # Adjust based on your network speed

def download_image(movie_id, url):
    """Download a single image and save it with the MovieID as the filename."""
    if pd.isna(url) or str(url).strip() in ['0', 'NaN', 'nan', '']:
        return False

    save_path = os.path.join(SAVE_DIR, f"{movie_id}.jpg")
    
    # Skip if already downloaded
    if os.path.exists(save_path):
        return True

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        return False

def main():
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    tasks = []
    print(f"Starting download with {MAX_THREADS} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for _, row in df.iterrows():
            movie_id = row['MovieID']
            url = row['Poster_URL']
            tasks.append(executor.submit(download_image, movie_id, url))
            
        # Optional: Simple progress tracking
        completed = 0
        total = len(tasks)
        for future in concurrent.futures.as_completed(tasks):
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed} / {total} downloaded or checked.")

    print("✅ All downloads completed!")

if __name__ == "__main__":
    main()