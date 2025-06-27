# 🎬 MovieSentiment AI

MovieSentiment AI is an intelligent movie review and recommendation system that blends **deep learning (LSTM)** with **natural language processing (NLP)** to analyze user-written reviews and classify them as **Positive** or **Negative**. Based on these sentiments, it tracks and suggests the best-rated movies in each genre. 🎯

---

## 🚀 Features

- 🧠 **LSTM-based Sentiment Analysis** for movie reviews.
- ✍️ **User Review Logging** with genre tracking.
- 📊 **Review Summary Dashboard**.
- 🎥 **Top 5 Movie Recommendations** by genre based on public sentiment.
- 🎨 Built using **Streamlit**, **TensorFlow**, and **NLTK**.

---

## 🛠 Tech Stack

- Python 🐍  
- Streamlit 🎨  
- TensorFlow & Keras 🧠  
- NLTK (Natural Language Toolkit) 📚  
- Pandas & NumPy 📊  
- Regex, Pickle, CSV 📁  

---

## 📂 File Structure

```
📁 MovieSentimentAI/
├── movie_sentiment.py        # 🧠 Run this to train the model and save tokenizer
├── movie.py                  # 🚀 Streamlit frontend app (main)
├── requirements.txt          # 📦 Dependencies
└── Updated_Review_CSV.csv    # 📊 Auto-generated after reviews
```

---

## ⚙️ Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/MovieSentimentAI.git
   cd MovieSentimentAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training script once to create model and tokenizer:**
   ```bash
   python movie_sentiment.py
   ```
   ✅ This generates:
   - `movie_sentiment.keras`
   - `movie_tokenizer.pickle`

4. **Start the Streamlit app:**
   ```bash
   streamlit run movie.py
   ```

---

## 🧪 How It Works

- Reviews are preprocessed using NLTK: stopword removal, lemmatization, regex cleaning.
- Cleaned text is tokenized and padded before passing into an **LSTM model**.
- Based on prediction score:
  - ≥ 0.5 → **Positive** 👍  
  - < 0.5 → **Negative** 👎  
- Reviews are stored in a CSV file and used to update top recommendations per genre.

---

## 🧼 Example Input

> `"This movie was so emotional and made me cry!"`  
Returns: `Positive (Confidence: 0.84)`

---

## 📊 Tabs in the App

- `🏠 Home` – Overview and navigation.
- `🎥 Recommend Movie` – Top 5 movies by genre.
- `📝 Review Movie` – Submit and classify reviews.
- `📊 Review Summary` – View all reviews in a table.

---

## 💡 Future Improvements

- Add IMDb/TMDB API for automatic movie info.
- User authentication for personal recommendations.
- Visualization of review trends over time.

---

## 👤 Author

Made with ❤️ by **WhiteDev08**

---

## 📜 License

This project is licensed under the MIT License.


---

Dataset : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---
## 🧼 Streamlit 

![image](https://github.com/user-attachments/assets/8887c23c-40a1-4218-9e39-ffce37dd2a5c)

![image](https://github.com/user-attachments/assets/869e7ead-e517-4e3b-b32f-0bad5684cbcb)


