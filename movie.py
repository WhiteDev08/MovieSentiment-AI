import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
critical_terms = {"sad", "cry", "depressed", "hopeless", 'am'}
stop_words = stop_words - critical_terms
additional_stopwords = {'life', 'something', 'anything', 'aand', 'abt', 'ability', 'academic', 'able', 'account', 'advance'}
stop_words.update(additional_stopwords)

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words).strip()

@st.cache_resource
def load_resources():
    model = load_model('movie_sentiment.keras')
    with open('movie_tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()
max_length = 28

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

# Load or initialize review summary
csv_file = 'Updated_Review_CSV.csv'
if os.path.exists(csv_file):
    df_summary = pd.read_csv(csv_file)
else:
    df_summary = pd.DataFrame(columns=['Movie', 'Genre', 'Positive Reviews', 'Negative Reviews'])

# Sidebar navigation
st.sidebar.title("📂 Movie Recommendation Guide")
tab = st.sidebar.radio("Go to", ["🏠 Home", "🎥 Recommend Movie", "📝 Review Movie", "📊 Review Summary"])

# ----------------------- TAB 1: HOME -----------------------
if tab == "🏠 Home":
    st.title("🎬 Movie Review Sentiment System")
    st.markdown("""
        Welcome to the **Movie Sentiment Tracker**! 📽️🍿

        - 🎯 This tool allows users to give feedback (reviews) on movies.
        - 🧠 An LSTM-based deep learning model classifies the review as **Positive** or **Negative**.
        - 📈 Reviews are logged to provide recommendations and summaries.
        - 🎨 Built using **Streamlit**, **TensorFlow**, and **NLP preprocessing**.

        ---
        🔍 **Tabs:**
        - `Recommend Movie`: Get top 5 recommended movies by genre.
        - `Review Movie`: Submit your own review!
        - `Review Summary`: See all logged reviews by movie.
    """)

# ----------------------- TAB 2: RECOMMEND -----------------------
elif tab == "🎥 Recommend Movie":
    st.title("🎥 Recommend Movie by Genre")
    genres = df_summary['Genre'].dropna().unique().tolist()
    if not genres:
        st.info("No reviews added yet.")
    else:
        selected_genre = st.selectbox("Choose a genre:", genres)
        top_movies = df_summary[df_summary['Genre'] == selected_genre]\
            .sort_values(by='Positive Reviews', ascending=False)\
            .head(5)
        if top_movies.empty:
            st.warning("No movies found in this genre.")
        else:
            st.subheader(f"Top 5 Movies in '{selected_genre}' Genre:")
            for i, row in top_movies.iterrows():
                st.markdown(f"🎬 **{row['Movie']}** — 👍 {row['Positive Reviews']} | 👎 {row['Negative Reviews']}")

# ----------------------- TAB 3: REVIEW -----------------------
elif tab == "📝 Review Movie":
    st.title("📝 Review a Movie")

    movie_name = st.text_input("Enter Movie Name:")
    review = st.text_area("Write your review here:")
    genre = st.selectbox("Select Genre:", ["Action", "Drama", "Comedy", "Sci-Fi", "Romance"])

    if st.button("Submit Review"):
        if not movie_name or not review or not genre:
            st.warning("Please fill in all fields.")
        else:
            sentiment, prob = predict_sentiment(review)
            normalized_movie = movie_name.strip().lower()
            df_summary['Movie_lower'] = df_summary['Movie'].str.lower()

            if normalized_movie in df_summary['Movie_lower'].values:
                idx = df_summary[df_summary['Movie_lower'] == normalized_movie].index[0]
                if sentiment == "Positive":
                    df_summary.at[idx, 'Positive Reviews'] += 1
                else:
                    df_summary.at[idx, 'Negative Reviews'] += 1
            else:
                new_row = {
                    'Movie': movie_name.strip(),
                    'Genre': genre,
                    'Positive Reviews': 1 if sentiment == "Positive" else 0,
                    'Negative Reviews': 1 if sentiment == "Negative" else 0
                }
                df_summary = pd.concat([df_summary, pd.DataFrame([new_row])], ignore_index=True)

            df_summary.drop(columns=['Movie_lower'], inplace=True)
            df_summary.to_csv(csv_file, index=False)

            st.success(f"Sentiment: {sentiment} (Confidence: {prob:.2f})")

            if sentiment == "Negative":
                    st.markdown("---")
                    st.subheader(f"💡 You might enjoy these top-rated '{genre}' movies:")

                    top_genre_movies = df_summary[df_summary['Genre'] == genre]\
                        .sort_values(by='Positive Reviews', ascending=False)\
                        .head(3)

                    if not top_genre_movies.empty:
                        for i, row in top_genre_movies.iterrows():
                            st.markdown(f"🎬 **{row['Movie']}** — 👍 {row['Positive Reviews']} | 👎 {row['Negative Reviews']}")
                    else:
                        st.info("No recommendations available yet for this genre.")


# ----------------------- TAB 4: SUMMARY -----------------------
elif tab == "📊 Review Summary":
    st.title("📊 Review Summary")
    if df_summary.empty:
        st.info("No reviews yet.")
    else:
        st.dataframe(df_summary)
