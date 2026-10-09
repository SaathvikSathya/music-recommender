# JAM! - Music Recommender

JAM! is a music recommendation app built with Python and Streamlit. Pick five songs you like, listen to recommended tracks, and give feedback by liking or skipping songs. The app uses your feedback to update its recommendations as you go.

It uses listening history from the Last.fm 1K dataset to find tracks based on user listening patterns. It also includes song previews and album artwork when available, along with a visual map of your music taste.

## Features

- Search for songs and pick five to get started.
- Get personalized song recommendations based on your selections and feedback.
- Like or skip tracks to influence what gets recommended next.
- View song previews and album artwork when available.
- Explore a visual map of your selected, liked, skipped, and upcoming tracks.
- Track your listening session with a history of your choices.

## Built With

- Python
- Streamlit
- Pandas and NumPy
- SciPy
- scikit-learn
- Plotly
- Apple iTunes Search API for song previews and artwork

## Dataset

This project uses the Last.fm 1K dataset, which contains user listening records, timestamps, artist names, and track names.

The dataset is not included in this repository because of its large file size. Download the dataset separately and place the required TSV file at:

`lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv`

## Run Locally

Install the required libraries:

```bash
pip install streamlit pandas numpy requests scipy scikit-learn plotly
```

Run the app:

```bash
streamlit run app.py
```

Replace `app.py` with the name of the Python file containing the app, if different.

The app loads the dataset and builds its recommendation model when it starts. The first load may take a little time because the dataset contains millions of listening records.

## About the Project

I built JAM! to explore music recommendations in an interactive way, combining listening data with feedback from users. The goal is to make discovering the next song feel more like building a playlist than browsing a list of recommendations.
