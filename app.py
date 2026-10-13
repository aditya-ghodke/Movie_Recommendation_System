import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load movie data
movies = pd.read_csv("movies.csv")


# Combine genre and description
movies["features"] = movies["genre"] + " " + movies["description"]


# Convert text into numbers
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(movies["features"])


# Calculate similarity
similarity = cosine_similarity(feature_matrix)


# Recommendation function
def recommend(movie_name):

    movie_index = movies[movies["title"] == movie_name].index[0]

    similarity_scores = list(enumerate(similarity[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:6]:
        recommendations.append(movies.iloc[index]["title"])

    return recommendations


# -------------------------------
# Website
# -------------------------------

st.title("🎬 Movie Recommendation System")

st.write("Find movies similar to your favorite movie!")


# Movie selection
movie_name = st.selectbox(
    "Select a movie:",
    movies["title"]
)


# Recommendation button
if st.button("Recommend Movies"):

    recommendations = recommend(movie_name)

    st.subheader("Recommended Movies:")

    for movie in recommendations:
        st.write("⭐", movie)
        