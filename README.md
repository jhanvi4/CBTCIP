# CBTCIP
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

# Load the dataset
df = pd.read_csv('Spam Email Detection.csv')
df.head()
df.shape
df.columns
df.drop_duplicates(inplace = True)
print(df.shape)
print(df.isnull().sum())

#Unamed 2, 3, 4 Columns will be Dropped because they conatin more than 90% Null values.
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df.head()

#!pip install nltk

#nltk.download("stopwords")

#Basic Pre-Processing

#Lower Case conversion:

df['v2'] = df['v2'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['v2'].head()
#Removal of Punctuation:
df['v2'] = df['v2'].str.replace('[^\w\s]','')
df['v2'].head()

#Removal of StopWords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['v2'] = df['v2'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['v2'].head()

#Stemming -refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['v2'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
X =  df.drop('v1', axis=1)
y = df['v1']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a CountVectorizer instance
vectorizer = CountVectorizer()
# Fit and transform the training data (X_train)
X_train_vectorized = vectorizer.fit_transform(X_train['v2'])

# Transform the test data (X_test)
X_test_vectorized = vectorizer.transform(X_test['v2'])

##Naive Bayes Classifier
#create and train the Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)
print(classifier.predict(X_train_vectorized))
print(ytrain.values)

#Evaluating the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred_train = classifier.predict(X_train_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_train, y_pred_train)
conf_matrix = confusion_matrix(y_train, y_pred_train)
classification_rep = classification_report(y_train, y_pred_train)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
#print the predictions
print(classifier.predict(X_test_vectorized))
#print the actual values
print(ytest.values)
#Evaluating the model on the test data set
y_pred_test = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)
classification_rep = classification_report(y_test, y_pred_test)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

##RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=200, random_state=0)
RF_model.fit(X_train_vectorized, y_train)
#Evaluating the model on the training data set

y_pred_rf_train = RF_model.predict(X_train_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_train, y_pred_rf_train)
conf_matrix = confusion_matrix(y_train, y_pred_rf_train)
classification_rep = classification_report(y_train, y_pred_rf_train)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
#Evaluating the model on the test data set
y_pred_rf_test = RF_model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rf_test)
conf_matrix = confusion_matrix(y_test, y_pred_rf_test)
classification_rep = classification_report(y_test, y_pred_rf_test)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)











