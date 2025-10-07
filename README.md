# 🎬 Movie Review Sentiment Analyzer

A machine learning–based web application that analyzes the sentiment of movie reviews and classifies them as Positive or Negative using Logistic Regression and TF-IDF vectorization

---

## 🧠 Key Learning Goals

- **Data Preprocessing**: Clean and prepare raw text data for modeling.
- **Model Training**: Build and evaluate a classic NLP classifier.
- **Model Persistence**: Save trained models for future use.
- **Web App Development**: Create an interactive UI for predictions.

---

## 📦 Tech Stack

- **Python**
- **scikit-learn**
- **pandas**
- **joblib**
- **Streamlit**

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `train.py` | Trains the sentiment classifier and saves model/vectorizer |
| `app.py` | Streamlit web app for user interaction |
| `model.joblib` | Saved sentiment classification model |
| `vectorizer.joblib` | Saved TF-IDF vectorizer |
| `requirements.txt` | List of required Python packages |

---

## 📊 Model Details

- **Dataset**: IMDB 50K Movie Reviews
- **Vectorizer**: TF-IDF (`TfidfVectorizer`)
- **Classifier**: Logistic Regression
- **Accuracy**: ~91% on test set

---

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/tulika105/Movie-Review-Sentiment-Analysis.git
cd Movie-Review-Sentiment-Analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py
```
---

## Live Demo

You can also try the app online without installing anything: 👉 https://2gjlrxhm8ypisyrdvgwnyu.streamlit.app/

---

## 📸 Screenshots

✅ Positive Review Prediction

<img width="1909" height="1070" alt="Positive review" src="https://github.com/user-attachments/assets/ab40daec-1e5d-4f98-92e7-434360ff8772" />

---

❌ Negative Review Prediction

<img width="1915" height="1072" alt="Negative review" src="https://github.com/user-attachments/assets/8615e3bb-173e-4985-a1be-cb81f0dda524" />

