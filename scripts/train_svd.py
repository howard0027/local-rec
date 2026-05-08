import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pickle
import os
import time

# --- Configuration ---
RATINGS_PATH = "./data/ml-1m/ratings.dat"
OUTPUT_DIR = "./data/processed/"
LATENT_DIM = 50  # Number of latent dimensions (hyperparameter)

def train_svd():
    print("🚀 Step 1: Loading MovieLens 1M ratings data...")
    start_time = time.time()
    
    ratings = pd.read_csv(RATINGS_PATH, sep='::', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], 
                          engine='python')
    
    print("🧠 Step 2: Building Sparse User-Item Matrix with Mean Centering...")
    user_ids = ratings['UserID'].unique()
    movie_ids = ratings['MovieID'].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    
    idx2user = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}

    # --- Mean Centering Logic ---
    global_mean = ratings['Rating'].mean()
    print(f"   -> Global Mean Rating: {global_mean:.4f}")
    ratings['Centered_Rating'] = ratings['Rating'] - global_mean

    row = ratings['UserID'].map(user2idx).values
    col = ratings['MovieID'].map(movie2idx).values
    data = ratings['Centered_Rating'].values

    num_users = len(user_ids)
    num_movies = len(movie_ids)
    R = csr_matrix((data, (row, col)), shape=(num_users, num_movies))
    
    print(f"⚙️ Step 3: Performing Truncated SVD (dim={LATENT_DIM})...")
    svd = TruncatedSVD(n_components=LATENT_DIM, random_state=42)
    
    user_embeddings = svd.fit_transform(R)
    movie_embeddings = svd.components_.T
    
    print("💾 Step 4: Saving models and mappings to disk...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, 'user_embeddings.npy'), user_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'movie_embeddings.npy'), movie_embeddings)
    
    # Save global_mean inside mappings for prediction reconstruction
    mappings = {
        'user2idx': user2idx,
        'movie2idx': movie2idx,
        'idx2user': idx2user,
        'idx2movie': idx2movie,
        'global_mean': global_mean
    }
    with open(os.path.join(OUTPUT_DIR, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)

    elapsed = time.time() - start_time
    print(f"✅ Training complete in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    train_svd()