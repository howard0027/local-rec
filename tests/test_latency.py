import sys
import os
import time
import pandas as pd

# Add parent directory to path so we can import ir_engine from the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir_engine.search import TrustRecEngine

def run_latency_tests():
    print("🚀 Loading data and initializing engine...")
    movies_df = pd.read_csv("./data/processed/movies_enriched.csv")
    
    engine = TrustRecEngine(movies_df, svd_dir="./data/processed/")

    # Create dummy user ratings to trigger the Folding-in matrix projection
    dummy_ratings = pd.DataFrame([
        {'UserID': 99999, 'MovieID': 1, 'Rating': 5, 'Timestamp': 123},
        {'UserID': 99999, 'MovieID': 260, 'Rating': 5, 'Timestamp': 123},
        {'UserID': 99999, 'MovieID': 3114, 'Rating': 4, 'Timestamp': 123}
    ])

    print("\n⏳ Starting Latency Tests (Measured in milliseconds)...\n")

    # Test 1: Pure Text Search (BM25 only, no SVD projection)
    start = time.perf_counter()
    engine.hybrid_search(query="space adventure", alpha=1.0, mmr_lambda=1.0, top_k=20)
    t1 = (time.perf_counter() - start) * 1000
    print(f"Test 1: Pure Text Search (Alpha=1.0, No MMR)          -> {t1:.2f} ms")

    # Test 2: Hybrid Search with Folding-in (SVD enabled, No MMR)
    start = time.perf_counter()
    engine.hybrid_search(query="space adventure", user_ratings=dummy_ratings, alpha=0.5, mmr_lambda=1.0, top_k=20)
    t2 = (time.perf_counter() - start) * 1000
    print(f"Test 2: Hybrid Search + Folding-in (Alpha=0.5, No MMR)  -> {t2:.2f} ms")

    # Test 3: Full Pipeline (Folding-in + Hybrid + MMR Diversity)
    start = time.perf_counter()
    engine.hybrid_search(query="space adventure", user_ratings=dummy_ratings, alpha=0.5, mmr_lambda=0.7, top_k=20)
    t3 = (time.perf_counter() - start) * 1000
    print(f"Test 3: Full Pipeline (Folding-in + Hybrid + MMR=0.7) -> {t3:.2f} ms")

    print("\n✅ Latency Tests Completed.")

if __name__ == "__main__":
    run_latency_tests()