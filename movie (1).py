import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to convert JSON-like strings into usable lists of names
def extract_names(data_str):
    if pd.isna(data_str):
        return ""
    try:
        parsed_data = ast.literal_eval(data_str)
        return " ".join([item["name"] for item in parsed_data])
    except (ValueError, KeyError, TypeError):
        return ""

# Load dataset
file_path = "C:\\Users\\masud\\Downloads\\Movie_rs_pp\\tmdb_5000_credits.csv"
data = pd.read_csv(file_path)

# Preprocess dataset
data['cast_names'] = data['cast'].apply(extract_names)
data['crew_names'] = data['crew'].apply(extract_names)
data['combined_features'] = data['title'] + " " + data['cast_names'] + " " + data['crew_names']

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(data['combined_features'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

# Page Configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide", page_icon="🎥")

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    .title-banner {
        background-color: #1f1f1f;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        color: #E0E0E0;
        font-family: Arial, sans-serif;
    }
    .movie-box {
        width: 75%;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px auto;
        border-radius: 10px;
        background-color: #2a2a2a;
        color: #FFFFFF;
        font-family: Arial, sans-serif;
        transition: transform 0.3s;
        position: relative;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .movie-box:hover {
        transform: scale(1.05);
        border-color: #555;
    }
    .movie-box:before {
        content: "";
        display: block;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #FF6F61, #6B8E23);
        position: absolute;
        top: 0;
        left: 0;
        border-radius: 10px 10px 0 0;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #AAAAAA;
        font-family: Arial, sans-serif;
    }
    .stSelectbox [data-baseweb="select"] {
        width: 50%;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Banner
st.markdown(
    "<div class='title-banner'><h1>🎬 Movie Recommendation System 🎥🍿</h1></div>",
    unsafe_allow_html=True,
)

# Dropdown Suggestions
movie_titles = data['title'].dropna().sort_values().unique().tolist()
selected_movie = st.selectbox("Select a movie to get recommendations 🎬👇:", movie_titles)

if selected_movie:
    # Ensure the movie exists in the dataset
    movie_idx = data[data['title'].str.lower() == selected_movie.lower()].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [score[0] for score in sorted_scores[1:6]]

    # Display Recommendations
    st.subheader(f"Recommendations for '{selected_movie}': 🎥✨")
    for idx, movie in enumerate(data.iloc[top_indices]['title'], start=1):
        st.markdown(f"<div class='movie-box'>🎬 {idx}. {movie} 🍿</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>© 2024 Movie Recommendation System | CSE Project 🎓</div>",
    unsafe_allow_html=True,
)
