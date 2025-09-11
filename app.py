import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv
import os

# -------------------------
# Load TMDB API key
# -------------------------
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    st.error("TMDB API key not found. Add it to your .env file.")
    st.stop()

# -------------------------
# Load Sentiment Models
# -------------------------
log_reg = joblib.load("logistic_model.pkl")
tfidf_sentiment = joblib.load("tfidf_vectorizer.pkl")

# -------------------------
# Load Recommendation Data
# -------------------------
tfidf_rec = joblib.load("tfidf_vectorizer_Rec.pkl")
vectors_rec = joblib.load("vectors_Rec.pkl")
similarity = cosine_similarity(vectors_rec)

data = pd.read_csv("movie_data_processed.csv")  # must have 'title' and 'tags'
titles_set = set(data['title'].values)  # faster lookup

# -------------------------
# Helper Functions
# -------------------------
def predict_sentiment(text):
    X = tfidf_sentiment.transform([text])
    pred = log_reg.predict(X)[0]
    prob = log_reg.predict_proba(X)[0]
    sentiment = "Positive 😀" if pred == 1 else "Negative 😡"
    confidence = max(prob)
    return sentiment, confidence

def get_tmdb_movie_details(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return movie_name, "Error fetching data from TMDB.", None

        results = response.json().get("results", [])
        if not results:
            return movie_name, "No overview available.", None

        result = results[0]
        title = result.get("title", movie_name)
        overview = result.get("overview", "No overview available.")
        poster_path = result.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return title, overview, poster_url
    except Exception as e:
        return movie_name, f"Error: {str(e)}", None

def recommend(movie_name):
    if movie_name not in titles_set:
        return []

    movie_index = data[data['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[movie_index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:9]  # top 8

    recommended_movies = []
    for i in movies_list:
        idx = i[0]
        title = data.iloc[idx]['title']
        title, overview, poster_url = get_tmdb_movie_details(title)
        recommended_movies.append({
            "title": title,
            "overview": overview,
            "poster_url": poster_url
        })
    return recommended_movies

# -------------------------
# Streamlit UI
# -------------------------

# -------------------------
# Section 1: Movie Recommendation (moved up)
# -------------------------
st.header("🎥 Movie Recommendation System")
movie_list = data['title'].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)
    if not recommendations:
        st.warning("Movie not found in database or no recommendations available.")
    else:
        st.markdown("### Recommended Movies")

        # Display in grid layout (3 per row)
        for i in range(0, len(recommendations), 3):
            cols = st.columns(3)
            for j, rec in enumerate(recommendations[i:i+3]):
                with cols[j]:
                    if rec['poster_url']:
                        st.image(rec['poster_url'], use_container_width=True)
                    st.subheader(rec['title'])
                    with st.expander("Overview"):
                        st.write(rec['overview'])

st.markdown("---")

# -------------------------
# Section 2: Sentiment Analysis (heading removed)
# -------------------------
user_review = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        sentiment, confidence = predict_sentiment(user_review)
        st.success(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")
