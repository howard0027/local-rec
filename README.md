# A Personalized Local Video Recommendation System

This project is an interactive discovery engine that combines traditional keyword search with personalized movie recommendations. It uses BM25 for text retrieval and SVD for latent taste modeling.

## 1. Setup and Installation

Follow these steps to set up the environment on your local machine.

### Clone the Project
First, download the project files from the repository:
```bash
git clone https://github.com/howard0027/local-rec
cd local-rec
```

### Create a Virtual Environment
It is recommended to use a virtual environment to keep the dependencies organized:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Required Packages
Install all necessary libraries using pip:
```bash
pip install -r requirements.txt
```

This is a **one-time effort**.

---

## 2. TMDB API Key Setup

The metadata enrichment step requires a TMDB API key.

### 2.1 Apply for a TMDB API Key
1. Create an account at [TMDB](https://www.themoviedb.org/).
2. Open your account settings and go to the API section.
3. Request API access and generate a **V3 API key**.
4. Keep the key private. Do not share it in code, screenshots, or commits.

### 2.2 Configure Your API Key Locally

You can choose one of the following methods.

#### Option A (Recommended): Set environment variable in terminal
```bash
# PowerShell
$env:TMDB_API_KEY="your_tmdb_v3_api_key_here"

# Windows CMD
set TMDB_API_KEY=your_tmdb_v3_api_key_here

# macOS/Linux
export TMDB_API_KEY="your_tmdb_v3_api_key_here"
```

#### Option B: Use a local `.env` file
Copy `.env.example` to `.env`, then set your key:
```bash
TMDB_API_KEY=your_tmdb_v3_api_key_here
```

The script `scripts/enrich_metadata.py` reads `TMDB_API_KEY` from the environment and falls back to `.env` automatically.

### 2.3 Verify Your Configuration
Run:
```bash
python3 scripts/enrich_metadata.py
```
- If the key is configured, the script starts processing movie metadata.
- If the key is missing, it stops immediately with a `TMDB_API_KEY is not set` error message.

### 2.4 Security and Incident Response
- Never commit `.env` or real API keys.
- If a key was exposed before, revoke/regenerate it in TMDB dashboard immediately.
- If the exposed key was pushed to a remote repo, rotate the key first, then clean git history if needed.

---

## 3. Data Preparation (Offline Phase)

Before running the application, you must prepare the dataset, fetch movie metadata, and train the recommendation model. This is a **one-time effort**.

### Step 1: Unzip the MovieLens 1M Dataset
Due to licensing and file size constraints, the raw dataset is not included in this repository.
```bash
cd data && unzip ml-1m.zip
cd ..
```


### Step 2: Enrich Movie Metadata
Run this script to connect the raw MovieLens dataset with the TMDB API. It will fetch movie URLs and summaries, saving the output as `movies_enriched.csv`:
```bash
python3 scripts/enrich_metadata.py
```
If `TMDB_API_KEY` is missing, the script will fail immediately with setup instructions.

### Step 3: Download Movie Posters
To display movie images in the UI, run the poster downloader script. This will use multithreading to download images based on the URLs fetched in the previous step:
```bash
python3 scripts/download_poster.py
```
*(Note: This step may take a few minutes depending on your network speed.)*

### Step 4: Train the SVD Model
Run this script to train the latent factor model. It handles the "Global Mean Centering" to ensure accurate recommendations even for unrated movies:
```bash
python3 scripts/train_svd.py
```

---

## 4. Running the Application (Online Phase)

After the data is prepared and models are trained, you can start the interactive interface.

### Launch the UI
Run the following command to start the Streamlit application:
```bash
streamlit run app.py
```
The system will automatically open a new tab in your web browser at `http://localhost:8501`. You can now explore the homepage, search for movies, and adjust the alpha and lambda sliders to control the algorithm.

---

## 5. Running Evaluation Tests

We provide test scripts to evaluate the accuracy and speed of the system. You can find them in the `tests/` folder.

### Evaluate Recommendation Accuracy (RMSE)
To see how well the SVD model predicts movie ratings, run:
```bash
python3 tests/test_rmse.py
```
This script will output the Mean Squared Error (MSE) and Root Mean Square Error (RMSE). An RMSE below 1.0 is considered a strong result.

### Evaluate System Latency
To measure how fast the engine responds to search and recommendation requests, run:
```bash
python3 tests/test_latency.py
```
The results will show the processing time (in milliseconds) for different scenarios, including pure text search and the full hybrid pipeline.

---

## Project Structure Overview
* `data/`: Contains raw datasets, processed CSV files, and downloaded posters (Not included in repo).
* `scripts/`: Data preprocessing, API fetching, and offline model training.
* `ir_engine/`: Core logic for hybrid search and MMR reranking.
* `tests/`: Scripts for evaluating system performance.
* `app.py`: The main Streamlit application file.