# ***Building a Sentiment Analysis Web App***

"""**1) Install libraries**"""

!pip install pandas scikit-learn streamlit joblib

"""**2) Import necessary packages**"""

# Import essential libraries for data processing, machine learning, and file handling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

"""**3) Load dataset**"""

# Load the IMDB movie reviews dataset as a pandas DataFrame
df = pd.read_csv("/content/IMDB Dataset.csv", engine='python')
df.head()

"""**4) Display shape and features**"""

# Display shape and columns to understand dataset dimensions and features
df.shape, df.columns

"""**5) Display sentiment count**"""

# View the distribution of sentiment labels in the dataset (positive/negative)
df['sentiment'].value_counts()

"""**6) Perform text cleaning**"""

def clean_text(text):
    """
    Cleans raw review text by:
    - Converting to lowercase
    - Removing HTML tags
    - Retaining only alphabetic characters and spaces
    - Collapsing multiple spaces
    """
    
    # Apply cleaning function to all review texts
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)          # remove HTML
    text = re.sub(r'[^a-z\s]', ' ', text)       # keep letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()    # collapse whitespace
    return text
df['review'] = df['review'].apply(clean_text)

"""**7) Display reviews after cleaning**"""

# Display top 10 reviews 
df['review'].head(10)

"""**8) Separating Data for Modeling**"""

# Separate the features (reviews) and labels (sentiment) for modeling
X = df['review']
y = df['sentiment']

"""**9) Preparing Training & Testing Sets**"""

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.2, random_state = 42)

"""**10) TF-IDF Vectorization**"""

# Convert text data into numerical representations using TF-IDF vectorization. Using max_features=20000 and bigrams for richer context representation
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

"""**11) Train Logistic Regression Model**"""

# Train a logistic regression classifier for binary sentiment prediction
model = LogisticRegression()
model.fit(X_train_vector, y_train)

"""**12) Model Evaluation**"""

# Evaluate model using accuracy, classification report, and confusion matrix. Compare predictions on the test set with ground truth
y_pred = model.predict(X_test_vector)
accuracy = accuracy_score(y_pred, y_test)
print('Training Score :', model.score(X_train_vector, y_train))
print('Accuracy :', accuracy)

print('Confusion Matrix:\n', confusion_matrix(y_pred, y_test))
print('\nclassification_report:\n', classification_report(y_pred, y_test))

"""**13) Test with custom reviews**"""

# Test model on a few custom movie review strings to validate its predictions, show predicted sentiment and model's confidence
# Sample reviews
test_reviews = [
    "I absolutely loved this movie! The story and performances were amazing.",
    "The film was boring and way too long. I wouldn’t recommend it.",
    "Not bad at all, but the ending felt rushed.",
    "Terrible acting and weak script. Waste of time."
]

# Clean and transform reviews
cleaned_reviews = [clean_text(r) for r in test_reviews]
X_test_vec = vectorizer.transform(cleaned_reviews)

# Make predictions
preds = model.predict(X_test_vec)
probs = model.predict_proba(X_test_vec)

# Display results
for i, review in enumerate(test_reviews):
    sentiment = preds[i]
    confidence = np.max(probs[i]) * 100
    print(f"\nReview: {review}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")

"""**13) Save Model and Vectorizer**"""

# Save the trained model and vectorizer to disk with joblib for use in deployment
# Save model
model_path = joblib.dump(model, "model.joblib")
print("Model saved to:", model_path)

# Save vectorizer
vectorizer_path = joblib.dump(vectorizer, "vectorizer.joblib")
print("Vectorizer saved to:", vectorizer_path)
