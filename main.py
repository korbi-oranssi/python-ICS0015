import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


# read in the dataset. coma separated, ignore the fist column, first row is the header
df = pd.read_csv("news.csv", sep=",", usecols=["title", "text", "label"], quotechar='"', header=0)

# split data into train and test datasets. test portion = 20%
x_train, x_test, y_train, y_test = train_test_split(df['text'], df["label"], test_size=0.2)

# initialize Tfidf the vectorizer. ignore english stop words
tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')

# learn vocabulary and idf(Inverse Document Frequency) from the training set. return term-document matrix
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

# transform to the term-document matrix using vocabulary from the training set
tfidf_test = tfidf_vectorizer.transform(x_test)

"""
# this is just an example of how the word weights are assigned
first_document_weights = pd.DataFrame(tfidf_train[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
# print the first document
print(x_train.iloc[0])
# print word weights for the first document
print(first_document_weights.sort_values(by=["tfidf"], ascending=False))
"""

# as we have a relatively small dataset and SVM are more reliable than more simplistic classifiers
# I use support vector machines for training
classifier = LinearSVC(max_iter=70)

# train the classifier
classifier.fit(tfidf_train, y_train)

# assessing the restults. predict results using the trained model
predictions = classifier.predict(tfidf_test)

# evaluate the score comparing actual and predicted values
score = accuracy_score(y_test, predictions)

# print the model accuracy
print(f'Model accuracy: {round(score*100,2)}%')