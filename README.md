# Restaurant Recommendation System

A restaurant recommender built on the [Yelp Open Dataset](https://www.yelp.com/dataset), with exploratory data analysis in a Jupyter notebook and an interactive **Streamlit** web app.

## How it works

The app combines two nearest-neighbour (KNN) models and merges their results:

1. **Category-based model:** each restaurant is one-hot encoded by its food categories (Pizza, Sushi, Breakfast & Brunch, …). The categories you pick are encoded the same way, and `NearestNeighbors` finds the 10 closest restaurants.
2. **Review-text model:** all reviews for each restaurant are cleaned (stop words and punctuation removed) and combined into one document, then vectorised with **TF-IDF**. Your own description (e.g. *"cozy place with spicy ramen and friendly staff"*) is vectorised the same way and matched to the 10 most similar restaurants.

The combined list is then filtered by the **city** and **minimum star rating** you choose.

The app can also show the latest reviews for any `business_id`.

## Project structure

```
.
├── Restaurant_recommendation_system.py      # Streamlit app
├── notebooks/
│   └── Restaurant_recommendations_with_EDA.ipynb   # EDA, preprocessing, model experiments
├── raw data/          # put the Yelp JSON files here (not tracked by git)
├── processed data/    # CSVs created by the notebook (not tracked by git)
└── requirements.txt
```

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only needed for the notebook
```

### 2. Get the data

Download the Yelp Open Dataset and put these two files in `raw data/`:

- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

The dataset is not included in this repo because of its size and Yelp's license terms.

### 3. Build the processed data

Open `notebooks/Restaurant_recommendations_with_EDA.ipynb` and run it from top to bottom. It filters the data to restaurants in the chosen cities (`cities_to_filter`, Toronto, Las Vegas and Phoenix by default), cleans the reviews, and saves four files to `processed data/`:

- `df_yelp_business.csv`
- `df_yelp_review.csv`
- `aggregated_reviews.csv`
- `result_df.csv`

> Which cities are available depends on the version of the Yelp dataset you download. If yours has different cities, change `cities_to_filter` in the notebook.

### 4. Run the app

```bash
streamlit run Restaurant_recommendation_system.py
```

In the sidebar, choose a city, one or more food categories and a minimum star rating, optionally describe what you're looking for (at least 5 words), then click **Submit**.

## Tech stack

Python · pandas · NumPy · scikit-learn (KNN, TF-IDF) · NLTK · spaCy · fuzzywuzzy · Matplotlib · Streamlit
