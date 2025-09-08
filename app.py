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

data = pd.read_csv("movie_data_processed.csv")  # should have at least 'title' and 'tags'

# -------------------------
# Helper Functions
# -------------------------
def predict_sentiment(text):
    X = tfidf_sentiment.transform([text])
    pred = log_reg.predict(X)[0]
    return "Positive 😀" if pred == 1 else "Negative 😡"

# Fetch movie details from TMDB
def get_tmdb_movie_details(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(url).json()
    try:
        result = response['results'][0]
        title = result.get('title', movie_name)
        overview = result.get('overview', "No overview available.")
        poster_path = result.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return title, overview, poster_url
    except:
        return movie_name, "No overview available.", None

def recommend(movie_name):
    if movie_name not in data['title'].values:
        return []
    
    movie_index = data[data['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[movie_index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]  # top 5 excluding the movie itself
    
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
st.set_page_config(page_title="Movie AI App", layout="wide")
st.title("🎬 Movie AI App")
st.markdown("Analyze review sentiments and get movie recommendations with posters!")

# Tabs
tab1, tab2 = st.tabs(["📝 Sentiment Analysis", "🎥 Movie Recommendation"])

# -------------------------
# Tab 1: Sentiment Analysis
# -------------------------
with tab1:
    st.header("Movie Review Sentiment Analysis")
    user_review = st.text_area("Enter your movie review here:")
    
    if st.button("Predict Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            result = predict_sentiment(user_review)
            st.success(f"Predicted Sentiment: {result}")

# -------------------------
# Tab 2: Movie Recommendation
# -------------------------
with tab2:
    st.header("Movie Recommendation System")
    movie_list = data['title'].values
    selected_movie = st.selectbox("Choose a movie:", movie_list)
    
    if st.button("Recommend Movies"):
        recommendations = recommend(selected_movie)
        if not recommendations:
            st.warning("Movie not found in database or no recommendations available.")
        else:
            for rec in recommendations:
                st.subheader(rec['title'])
                if rec['poster_url']:
                    st.image(rec['poster_url'], width=200)
                st.write(rec['overview'])
