# 🎬 Movie Review Sentiment Analyzer

A Streamlit web application that predicts whether a movie review is **Positive** or **Negative** using a Logistic Regression model trained on the IMDB 50K Movie Reviews Dataset. The model uses TF-IDF vectorization and achieves around 91% accuracy on the test set.

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

##  🌐 Live Demo

You can also try the app online without installing anything: 👉 https://2gjlrxhm8ypisyrdvgwnyu.streamlit.app/

---

## 📸 Screenshots

✅ Positive Review Prediction

<img width="1909" height="899" alt="Positive review" src="https://github.com/user-attachments/assets/8dec2637-be2e-4afa-a52d-77ea9206135f" />


---

❌ Negative Review Prediction

<img width="1915" height="885" alt="Negative review" src="https://github.com/user-attachments/assets/d672a21d-1d34-44e2-b9d8-ec9610abdbd6" />


