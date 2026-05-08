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

## 2. Data Preparation (Offline Phase)

Before running the application, you must prepare the dataset, fetch movie metadata, and train the recommendation model. This is a **one-time effort**.

### Step 1: Unzip the MovieLens 1M Dataset
Due to licensing and file size constraints, the raw dataset is not included in this repository.
`cd data && unzip ml-1m.zip`
`cd ..`

### Step 2: Enrich Movie Metadata
Run this script to connect the raw MovieLens dataset with the TMDB API. It will fetch movie URLs and summaries, saving the output as `movies_enriched.csv`:
```bash
python3 scripts/enrich_metadata.py
```

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

## 3. Running the Application (Online Phase)

After the data is prepared and models are trained, you can start the interactive interface.

### Launch the UI
Run the following command to start the Streamlit application:
```bash
streamlit run app.py
```
The system will automatically open a new tab in your web browser at `http://localhost:8501`. You can now explore the homepage, search for movies, and adjust the alpha and lambda sliders to control the algorithm.

---

## 4. Running Evaluation Tests

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