import numpy as np
import pandas as pd
import pickle
import os
import math

def calculate_rmse():
    print("🚀 Loading SVD Models and Datasets...")
    svd_dir = "./data/processed/"
    ratings_path = "./data/ml-1m/ratings.dat"

    try:
        user_embeddings = np.load(os.path.join(svd_dir, 'user_embeddings.npy'))
        movie_embeddings = np.load(os.path.join(svd_dir, 'movie_embeddings.npy'))
        with open(os.path.join(svd_dir, 'mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading SVD assets: {e}")
        print("Please ensure train_svd.py has been executed.")
        return

    user2idx = mappings['user2idx']
    movie2idx = mappings['movie2idx']
    global_mean = mappings.get('global_mean', 3.58) # Fallback to 3.58 if not found

    print("📊 Loading raw ratings for evaluation...")
    ratings = pd.read_csv(ratings_path, sep='::', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')

    sample_size = 10000
    test_sample = ratings.sample(n=sample_size, random_state=42)

    print(f"🧪 Evaluating RMSE on {sample_size} random samples...")
    
    squared_errors = []
    for _, row in test_sample.iterrows():
        uid = row['UserID']
        mid = row['MovieID']
        actual_rating = row['Rating']

        if uid in user2idx and mid in movie2idx:
            u_vec = user_embeddings[user2idx[uid]]
            m_vec = movie_embeddings[movie2idx[mid]]
            
            # Predict rating: Re-add global mean to the centered dot product
            predicted_rating = global_mean + np.dot(u_vec, m_vec)
            
            # Clip the prediction to realistic rating bounds (1 to 5 stars)
            predicted_rating = max(1.0, min(5.0, predicted_rating))
            
            squared_errors.append((actual_rating - predicted_rating) ** 2)

    if squared_errors:
        mse = sum(squared_errors) / len(squared_errors)
        rmse = math.sqrt(mse)
        print(f"\n📈 Results:")
        print(f"   -> MSE (Mean Squared Error): {mse:.4f}")
        print(f"   -> RMSE:                     {rmse:.4f}")
        
        if rmse < 1.0:
            print("   ✅ Excellent! RMSE is below 1.0.")
        else:
            print("   ⚠️ RMSE is a bit high. Consider increasing LATENT_DIM in train_svd.py.")
    else:
        print("❌ No valid user/movie pairs found in the sample.")

if __name__ == "__main__":
    calculate_rmse()