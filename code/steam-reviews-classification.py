import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import os
import json
import string

# data is in JSON Lines format, so we need a few steps in order to import it 

data_folder = os.path.join(os.getcwd(), "data")
data_files_names = os.listdir(data_folder)

data = []

for data_file_name in data_files_names:
    file = os.path.join(os.getcwd(), "data", data_file_name)
    with open(file, "r", encoding='utf-8') as f:
        lst = [json.loads(line) for line in f]
    data.extend(lst)

df = pd.DataFrame.from_records(data)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
lemmatized_reviews = []

for review in df["review"]:
    review = review.translate(str.maketrans('', '', string.punctuation)).lower()
    words_in_review = word_tokenize(review)
    filtered_list = []
    for word in words_in_review:
        if word not in stop_words:
            filtered_list.append(word)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_list]
    lemmatized_reviews.append(" ".join(lemmatized_words))

df["lemmatized_reviews"] = lemmatized_reviews

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["lemmatized_reviews"])
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = df["rating"].unique())
disp.plot()
plt.show()
