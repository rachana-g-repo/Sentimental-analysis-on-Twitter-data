import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv('Twitter_Data.csv')

df.head(5)
print(df.isnull().sum())

df.dropna(subset=['clean_text', 'category'], inplace=True)
print(df.isnull().sum())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned_text'] = df['clean_text'].apply(preprocess_text)

count_vectorizer = CountVectorizer(max_features=5000)
count_matrix = count_vectorizer.fit_transform(df['cleaned_text'])

X_train, X_test, y_train, y_test = train_test_split(count_matrix, df['category'], test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(count_vectorizer, vectorizer_file)  # Save the CountVectorizer object

with open('nb_classifier.pkl', 'wb') as classifier_file:
    pickle.dump(nb_classifier, classifier_file)  # Save the trained classifier