import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from ir_engine.search import TrustRecEngine

# Resolve Pandas fillna FutureWarning
pd.set_option('future.no_silent_downcasting', True)

st.set_page_config(page_title="TrustRec | Discovery Engine", page_icon="🎬", layout="wide")
st.markdown("""<style>.main { background-color: #0e1117; } .stButton>button { width: 100%; }</style>""", unsafe_allow_html=True)

# --- 1. Data & Engine Loading ---
@st.cache_data
def load_data():
    movies = pd.read_csv("./data/processed/movies_enriched.csv")
    base_ratings = pd.read_csv("./data/ml-1m/ratings.dat", sep='::', header=None,
                               names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')

    # [Dynamic User ID] Find max ID and assign the next one for local user
    base_max_id = base_ratings['UserID'].max()
    local_user_id = base_max_id + 1

    # Clean dirty data
    dirty_values = ['0', 'NaN', 'nan', 'none', '', 'None']
    movies['Poster_URL'] = movies['Poster_URL'].replace(dirty_values, pd.NA)
    
    if 'Year' not in movies.columns:
        movies['Year'] = movies['Original_Title'].str.extract(r'\((\d{4})\)$').astype(float)
        movies['Year'] = movies['Year'].fillna(1900).astype(int)

    # Load and merge local ratings
    my_ratings_path = "./data/processed/my_ratings.csv"
    if os.path.exists(my_ratings_path):
        my_ratings = pd.read_csv(my_ratings_path)
        if not my_ratings.empty:
            local_user_id = my_ratings['UserID'].iloc[0]
        ratings = pd.concat([base_ratings, my_ratings], ignore_index=True)
    else:
        ratings = base_ratings

    # Popularity stats
    stats = ratings.groupby('MovieID').agg(rating_count=('Rating', 'size'), rating_mean=('Rating', 'mean'))
    stats['Popularity_Score'] = stats['rating_count'] * stats['rating_mean']
    movies_with_pop = movies.merge(stats[['Popularity_Score']], on='MovieID', how='left')
    
    movies_with_pop['Popularity_Score'] = movies_with_pop['Popularity_Score'].fillna(0.0).infer_objects(copy=False)
    
    all_genres = set()
    for genres_str in movies['Genres'].dropna():
        all_genres.update(genres_str.split('|'))
    all_genres = sorted(list(all_genres))
    
    return movies_with_pop, ratings, all_genres, local_user_id

@st.cache_resource
def load_engine(_df):
    return TrustRecEngine(_df)

# --- CRUD Operations for Personal Ratings ---
def get_personal_ratings():
    file_path = "./data/processed/my_ratings.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty and 'Timestamp' in df.columns:
            df = df.sort_values(by='Timestamp', ascending=False)
            df = df.drop_duplicates(subset=['MovieID'], keep='first')
        return df
    return pd.DataFrame(columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])

def update_personal_rating(user_id: int, movie_id: int, rating: int):
    df = get_personal_ratings()
    df = df[df['MovieID'] != movie_id] 
    new_row = {'UserID': user_id, 'MovieID': movie_id, 'Rating': rating, 'Timestamp': int(time.time())}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("./data/processed/my_ratings.csv", index=False)
    st.session_state['unsynced_changes'] = True

def delete_personal_rating(movie_id: int):
    df = get_personal_ratings()
    df = df[df['MovieID'] != movie_id]
    df.to_csv("./data/processed/my_ratings.csv", index=False)
    
    # Auto-reset logic: If history becomes empty, silently clear caches
    if df.empty:
        st.session_state['unsynced_changes'] = False
        st.cache_resource.clear()
        st.cache_data.clear()
    else:
        st.session_state['unsynced_changes'] = True

# --- Event Callbacks (Instant UI Refresh) ---
def handle_rate_movie(user_id, movie_id, widget_key):
    val = st.session_state.get(widget_key)
    if val is not None:
        update_personal_rating(user_id, movie_id, val + 1)

def handle_remove_rating(movie_id):
    delete_personal_rating(movie_id)

# --- Image Rendering ---
def render_poster(movie_id, url):
    local_path = f"./data/posters/{movie_id}.jpg"
    white_path = f"./data/posters/0.jpg"
    
    if os.path.exists(local_path):
        st.image(local_path, width='stretch')
    elif pd.notna(url) and isinstance(url, str) and url.startswith("http"):
        try:
            st.image(url, width='stretch')
        except Exception:
            st.image(white_path if os.path.exists(white_path) else "https://via.placeholder.com/300x450?text=No+Poster", width='stretch')
    elif os.path.exists(white_path):
        st.image(white_path, width='stretch')
    else:
        st.image("https://via.placeholder.com/300x450?text=No+Poster", width='stretch')

# --- Initialize Resources ---
movies_df, ratings_df, all_available_genres, LOCAL_USER_ID = load_data()

if not movies_df.empty:
    engine = load_engine(movies_df)

    # --- 2. Sidebar: System Control ---
    with st.sidebar:
        st.title("⚙️ System State")
        personalization_enabled = st.toggle("✨ Enable Personalization", value=False, help="Activate SVD Collaborative Filtering.")
        active_user_id = LOCAL_USER_ID if personalization_enabled else None
        
        if not personalization_enabled:
            st.info("Current Mode: **Guest**.")
        else:
            st.success(f"Current Mode: **Personalized**.")
            st.markdown("---")
            st.subheader("🧠 Taste Profile")

            if st.session_state.get('unsynced_changes', False):
                st.warning("⚠️ Ratings changed! Sync to update recommendations.")
                btn_label = "🔴 Sync & Retrain Brain"
            else:
                btn_label = "🔄 Sync & Retrain Brain"
            
            if st.button(btn_label):
                with st.spinner("Re-computing taste coordinates..."):
                    st.session_state['unsynced_changes'] = False
                    st.cache_resource.clear() 
                    st.cache_data.clear()     
                st.success("Profile Updated!")
                st.rerun()

            with st.expander("📖 My Rating History", expanded=False):
                my_ratings = get_personal_ratings()
                if not my_ratings.empty:
                    my_ratings = my_ratings.sort_values(by='Timestamp', ascending=False)
                    for _, r_row in my_ratings.iterrows():
                        m_title = movies_df[movies_df['MovieID'] == r_row['MovieID']]['Clean_Title'].iloc[0]
                        r_score = int(r_row['Rating'])
                        col_text, col_btn = st.columns([4, 1])
                        col_text.write(f"**{m_title}**\n{'⭐' * r_score}")
                        col_btn.button("❌", key=f"del_hist_{r_row['MovieID']}", on_click=handle_remove_rating, args=(r_row['MovieID'],))
                else:
                    st.caption("No ratings yet.")
        
        st.markdown("---")
        st.subheader("Algorithmic Tuning")
        alpha = st.slider("⚖️ Text Match vs. Taste", 0.0, 1.0, 0.5, 0.1)
        mmr_lambda = st.slider("🔀 Diversity Control (MMR)", 0.0, 1.0, 0.7, 0.1)

    # --- 3. Movie Details Dialog ---
    @st.dialog("Movie Details", width="large")
    def show_movie_details(movie):
        col1, col2 = st.columns([1, 2])
        with col1:
            render_poster(movie['MovieID'], movie['Poster_URL'])
        with col2:
            st.header(f"{movie['Clean_Title']} ({int(movie['Year'])})")
            st.write(f"**Genres:** {movie['Genres']}")
            st.write(f"**Plot:** {movie['Overview']}")
            st.markdown(f"[🎥 Watch Trailer](https://www.youtube.com/results?search_query={movie['Clean_Title'].replace(' ', '+')}+trailer)")

        st.divider()
        st.subheader("💿 Similar Titles")
        similar_movies = engine.get_similar_movies(movie['MovieID'], top_k=5)
        if not similar_movies.empty:
            sim_cols = st.columns(5)
            for i, (_, sim_movie) in enumerate(similar_movies.iterrows()):
                with sim_cols[i]:
                    render_poster(sim_movie['MovieID'], sim_movie['Poster_URL'])
                    st.caption(sim_movie['Clean_Title'])

    # --- 4. Main Interface & Search Filters ---
    st.title("🎬 A Personalized Local Video Recommendation System")
    query = st.text_input("🔍 Search movies...", placeholder="e.g., Space adventure, Pixar animation...")
    
    with st.expander("🛠️ Advanced Search Filters", expanded=False):
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
        with f_col1:
            selected_genres = st.multiselect("Filter Genres", all_available_genres, default=[])
        with f_col2:
            db_min, db_max = int(movies_df['Year'].min()), int(movies_df['Year'].max())
            curr_year = datetime.now().year
            year_range = st.slider("Release Year Range", db_min, max(db_max, curr_year), (db_min, max(db_max, curr_year)))
        with f_col3:
            sort_option = st.selectbox("Sort By", ["Relevance", "Year (Newest)", "Year (Oldest)"])
            sort_mapping = {"Relevance": "relevance", "Year (Newest)": "year_desc", "Year (Oldest)": "year_asc"}
            sort_by_param = sort_mapping[sort_option]

    # --- 5. Content Rendering ---
    user_history_df = get_personal_ratings()

    # Shared UI Component for Movie Cards
    def render_movie_card(row, key_prefix, show_title=False):
        render_poster(row['MovieID'], row['Poster_URL'])
        
        if show_title:
            st.write(f"**{row['Clean_Title']}**")
            
        if personalization_enabled:
            existing_match = user_history_df[user_history_df['MovieID'] == row['MovieID']]
            if not existing_match.empty:
                current_stars = int(existing_match['Rating'].iloc[0])
                st.caption(f"**Rated:** {'⭐' * current_stars}")
            
            # ⭐️ Always show the feedback widget to allow updating
            widget_key = f"star_{key_prefix}_{row['MovieID']}"
            st.feedback("stars", key=widget_key, on_change=handle_rate_movie, args=(LOCAL_USER_ID, row['MovieID'], widget_key))
            
            if not existing_match.empty:
                st.button("🗑️ Remove", key=f"rm_{key_prefix}_{row['MovieID']}", on_click=handle_remove_rating, args=(row['MovieID'],))
                
        if st.button("Details", key=f"det_{key_prefix}_{row['MovieID']}"):
            show_movie_details(row)

    if query:
        st.subheader(f"Search Results for: '{query}'")
        results = engine.hybrid_search(
            query=query, 
            user_id=active_user_id, 
            user_ratings=user_history_df if personalization_enabled else None,
            alpha=alpha, 
            mmr_lambda=mmr_lambda, 
            top_k=20, 
            year_range=year_range, 
            genres_filter=selected_genres, 
            sort_by=sort_by_param
        )
        
        if results.empty:
            st.warning("No matches found. Try broadening your filters.")
        else:
            cols = st.columns(5)
            for idx, row in results.reset_index().iterrows():
                with cols[idx % 5]:
                    render_movie_card(row, f"search_{row['MovieID']}", show_title=True)
    else:
        seen_ids = set() 
        is_cold_start = personalization_enabled and user_history_df.empty

        if not personalization_enabled or is_cold_start:
            if is_cold_start:
                st.info("❄️ **Cold Start:** You are in Personalized Mode, but we need you to rate some movies first! Here are global trends for now.")
            else:
                st.subheader("🔥 Global Trending")
                
            shelf_genres = selected_genres[:3] if selected_genres else ['Action', 'Sci-Fi', 'Comedy']
            for shelf_idx, genre in enumerate(shelf_genres):
                st.markdown(f"### 🚀 Trending in **{genre}**")
                
                genre_candidates = movies_df[
                    (movies_df['Genres'].str.contains(genre, na=False)) & 
                    (~movies_df['MovieID'].isin(seen_ids)) &
                    (movies_df['Year'] >= year_range[0]) & (movies_df['Year'] <= year_range[1])
                ].copy()
                
                if not genre_candidates.empty:
                    candidates = genre_candidates.sort_values(by='Popularity_Score', ascending=False).head(50)
                    
                    max_pop = candidates['Popularity_Score'].max()
                    candidates['Total_Score'] = candidates['Popularity_Score'] / max_pop if max_pop > 0 else 0.0
                    
                    shelf_results = engine._apply_mmr(candidates, lambda_val=mmr_lambda, top_k=5)
                    
                    seen_ids.update(shelf_results['MovieID'].tolist())
                    cols = st.columns(5)
                    for idx, row in shelf_results.reset_index().iterrows():
                        with cols[idx % 5]:
                            render_movie_card(row, f"guest_{shelf_idx}_{row['MovieID']}")
                st.markdown("---")
        else:
            st.subheader("✨ Recommended For You")
            history_with_genres = user_history_df.merge(movies_df[['MovieID', 'Genres']], on='MovieID')
            shelf_genres = history_with_genres['Genres'].str.split('|').explode().value_counts().head(3).index.tolist()

            for shelf_idx, genre in enumerate(shelf_genres):
                st.markdown(f"### 📂 Because you like **{genre}**")
                genre_candidates = movies_df[
                    (movies_df['Genres'].str.contains(genre, na=False)) & 
                    (~movies_df['MovieID'].isin(seen_ids)) &
                    (movies_df['Year'] >= year_range[0]) & (movies_df['Year'] <= year_range[1])
                ].copy()
                
                shelf_engine = TrustRecEngine(genre_candidates)
                
                shelf_results = shelf_engine.hybrid_search(
                    query="", 
                    user_id=active_user_id, 
                    user_ratings=user_history_df,
                    alpha=alpha,           
                    mmr_lambda=mmr_lambda,
                    top_k=5
                )
                                
                if not shelf_results.empty:
                    seen_ids.update(shelf_results['MovieID'].tolist())
                    cols = st.columns(5)
                    for idx, row in shelf_results.reset_index().iterrows():
                        with cols[idx % 5]:
                            render_movie_card(row, f"user_{shelf_idx}_{row['MovieID']}")
                st.markdown("---")