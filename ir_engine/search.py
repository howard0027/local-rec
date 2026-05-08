import pandas as pd
import numpy as np
import os
import pickle
import difflib
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class TrustRecEngine:
    def __init__(self, df: pd.DataFrame, svd_dir: str = "./data/processed/"):
        self.df = df.copy()
        
        # Ensure string types to prevent TypeErrors with pure numeric titles
        self.df['Overview'] = self.df['Overview'].fillna("").astype(str)
        self.df['Clean_Title'] = self.df['Clean_Title'].fillna("").astype(str)
        self.df['Genres'] = self.df['Genres'].fillna("").astype(str)
        
        # Initialize BM25 (Lexical Search)
        self.corpus = (self.df['Clean_Title'] + " ") * 2 + (self.df['Genres'] + " ") * 2 + self.df['Overview']
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Pre-extract lowercase titles for faster fuzzy matching
        self.all_titles_lower = self.df['Clean_Title'].str.lower().tolist()
        
        # Load SVD Models (Latent Search)
        try:
            self.user_embeddings = np.load(os.path.join(svd_dir, 'user_embeddings.npy'))
            self.movie_embeddings = np.load(os.path.join(svd_dir, 'movie_embeddings.npy'))
            with open(os.path.join(svd_dir, 'mappings.pkl'), 'rb') as f:
                self.mappings = pickle.load(f)
            self.user2idx = self.mappings['user2idx']
            self.movie2idx = self.mappings['movie2idx']
            self.global_mean = self.mappings.get('global_mean', 3.58) # Fallback if missing
            self.svd_loaded = True
        except FileNotFoundError:
            self.svd_loaded = False
            self.global_mean = 3.58
            print("Warning: SVD models not found. Using BM25 only.")

    def _normalize(self, scores):
        if len(scores) == 0: return scores
        min_v, max_v = np.min(scores), np.max(scores)
        if max_v == min_v: return np.zeros_like(scores)
        return (scores - min_v) / (max_v - min_v)

    def infer_user_vector(self, user_ratings_df: pd.DataFrame):
        """Folding-in: Projects a user into latent space using weighted movie vectors."""
        if not self.svd_loaded or user_ratings_df.empty:
            return None
        
        mask = user_ratings_df['MovieID'].isin(self.movie2idx.keys())
        valid_ratings = user_ratings_df[mask]
        
        if valid_ratings.empty:
            return None
            
        vectors = []
        weights = []
        for _, row in valid_ratings.iterrows():
            m_idx = self.movie2idx[row['MovieID']]
            vectors.append(self.movie_embeddings[m_idx])
            # Center ratings based on true global mean to calculate shifting weight
            weights.append(row['Rating'] - self.global_mean)
            
        weights = np.array(weights)
        
        # Fallback for ZeroDivisionError: If weights perfectly cancel out
        if np.sum(weights) == 0:
            return np.mean(vectors, axis=0)
            
        return np.average(vectors, axis=0, weights=weights)

    def _apply_mmr(self, candidates_df: pd.DataFrame, lambda_val: float, top_k: int) -> pd.DataFrame:
        """Maximal Marginal Relevance (MMR) for diversity re-ranking."""
        if candidates_df.empty or lambda_val == 1.0:
            return candidates_df.head(top_k)
            
        mask = candidates_df['MovieID'].isin(self.movie2idx.keys())
        valid_cands = candidates_df[mask].copy()
        
        if valid_cands.empty:
            return candidates_df.head(top_k)

        candidate_idx = [self.movie2idx[mid] for mid in valid_cands['MovieID']]
        candidate_vectors = self.movie_embeddings[candidate_idx]
        scores = valid_cands['Total_Score'].values 
        
        selected_indices = []
        selected_vectors = []
        unselected_indices = list(range(len(valid_cands)))
        
        best_idx = np.argmax(scores)
        selected_indices.append(best_idx)
        selected_vectors.append(candidate_vectors[best_idx])
        unselected_indices.remove(best_idx)
        
        while len(selected_indices) < top_k and len(unselected_indices) > 0:
            unselected_vecs = candidate_vectors[unselected_indices]
            sim_matrix = cosine_similarity(unselected_vecs, np.array(selected_vectors))
            max_sim_to_selected = np.max(sim_matrix, axis=1)
            
            rel_scores = scores[unselected_indices]
            mmr_scores = (lambda_val * rel_scores) - ((1 - lambda_val) * max_sim_to_selected)
            
            best_unselected_idx = np.argmax(mmr_scores)
            real_idx = unselected_indices[best_unselected_idx]
            
            selected_indices.append(real_idx)
            selected_vectors.append(candidate_vectors[real_idx])
            unselected_indices.remove(real_idx)
            
        return valid_cands.iloc[selected_indices]

    def hybrid_search(self, query: str, user_id: int = None, user_ratings: pd.DataFrame = None, 
                      alpha: float = 0.5, mmr_lambda: float = 1.0, top_k: int = 10, 
                      year_range: tuple = None, genres_filter: list = None, sort_by: str = 'relevance',
                      use_prf: bool = True) -> pd.DataFrame:
        
        results = self.df.copy()
        
        # --- 1. BM25 & Fuzzy Search (with PRF) ---
        bm25_raw = np.zeros(len(self.df))
        if query.strip():
            query_lower = query.lower()
            tokenized_query = query_lower.split()
            
            # Pass 1: Initial Retrieval
            bm25_raw = self.bm25.get_scores(tokenized_query)
            
            close_matches = difflib.get_close_matches(query_lower, self.all_titles_lower, n=3, cutoff=0.7)
            fuzzy_mask = self.df['Clean_Title'].str.lower().isin(close_matches) if close_matches else None
            
            # Pass 2: Pseudo-Relevance Feedback (PRF)
            if use_prf and np.max(bm25_raw) > 0:
                top_indices = np.argsort(bm25_raw)[-3:][::-1]
                expansion_terms = []
                for idx in top_indices:
                    if bm25_raw[idx] > 0:
                        genres = self.df.iloc[idx]['Genres'].lower().replace('|', ' ').split()
                        expansion_terms.extend(genres)
                if expansion_terms:
                    expanded_query = tokenized_query + list(set(expansion_terms))
                    bm25_raw = self.bm25.get_scores(expanded_query)

            # Apply Fuzzy Matching Boost after PRF
            if fuzzy_mask is not None and fuzzy_mask.any():
                bm25_raw[fuzzy_mask] += np.max(bm25_raw) * 0.5 + 2.0 
                
        results['BM25_Score'] = self._normalize(bm25_raw)
        
        # --- 2. SVD Score (Integrated with Centering & Fallback) ---
        svd_raw = np.zeros(len(results))
        user_vector = None
        
        if self.svd_loaded:
            if user_id is not None and user_id in self.user2idx:
                u_idx = self.user2idx[user_id]
                user_vector = self.user_embeddings[u_idx] 
            elif user_ratings is not None and not user_ratings.empty:
                user_vector = self.infer_user_vector(user_ratings)
                
        if user_vector is not None:
            mask = results['MovieID'].isin(self.movie2idx.keys())
            if mask.any(): 
                valid_movie_ids = results.loc[mask, 'MovieID']
                m_indices = [self.movie2idx[m_id] for m_id in valid_movie_ids]
                movie_vectors = self.movie_embeddings[m_indices]
                # Re-add global mean to centered dot product to align scores
                svd_raw[mask] = self.global_mean + np.dot(movie_vectors, user_vector)
        else:
            # Guest / Cold Start Mode: Fallback to Global Popularity
            if 'Popularity_Score' in results.columns:
                svd_raw = results['Popularity_Score'].values
                
        results['SVD_Score'] = self._normalize(svd_raw)
        
        # --- 3. Hybrid Score ---
        results['Total_Score'] = (alpha * results['BM25_Score']) + ((1 - alpha) * results['SVD_Score'])
        
        if query.strip() and alpha > 0:
            results = results[results['Total_Score'] > 0]
            
        # --- 4. Apply Hard Filters ---
        if genres_filter:
            pattern = '|'.join(genres_filter)
            results = results[results['Genres'].str.contains(pattern, case=False, na=False)]
            
        if year_range and 'Year' in results.columns:
            results = results[(results['Year'] >= year_range[0]) & (results['Year'] <= year_range[1])]
            
        # --- 5. Sorting & MMR Diversity ---
        if sort_by == 'relevance':
            results = results.sort_values(by='Total_Score', ascending=False)
            if mmr_lambda < 1.0 and len(results) > 0:
                candidates = results.head(50)
                results = self._apply_mmr(candidates, lambda_val=mmr_lambda, top_k=top_k)
            else:
                results = results.head(top_k)
        elif sort_by == 'year_desc':
            results = results.sort_values(by=['Year', 'Total_Score'], ascending=[False, False]).head(top_k)
        elif sort_by == 'year_asc':
            results = results.sort_values(by=['Year', 'Total_Score'], ascending=[True, False]).head(top_k)
            
        return results

    def get_similar_movies(self, movie_id: int, top_k: int = 5) -> pd.DataFrame:
        """Find similar movies via Cosine Similarity in the SVD latent space."""
        if not self.svd_loaded or movie_id not in self.movie2idx:
            return pd.DataFrame()
            
        target_idx = self.movie2idx[movie_id]
        target_vector = self.movie_embeddings[target_idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vector, self.movie_embeddings).flatten()
        
        related_indices = sim_scores.argsort()[-(top_k + 1):][::-1]
        related_indices = [idx for idx in related_indices if idx != target_idx][:top_k]
        related_movie_ids = [self.mappings['idx2movie'][i] for i in related_indices]
        
        res_df = self.df[self.df['MovieID'].isin(related_movie_ids)].copy()
        res_df['MovieID'] = pd.Categorical(res_df['MovieID'], categories=related_movie_ids, ordered=True)
        return res_df.sort_values('MovieID')