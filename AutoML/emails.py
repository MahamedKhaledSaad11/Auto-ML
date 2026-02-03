import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import string
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from models import *
from models import train_ann_email, evaluate_ann
from sklearn.model_selection import train_test_split
df=pd.read_csv("emails.csv")

df["clean_text"] = df["text"].apply(clean_text)
df["tokens"] = df["clean_text"].apply(lambda x: word_tokenize(x))
wordnet_lemmatizer = WordNetLemmatizer()
df["lemmatized"] = df["tokens"].apply(lambda tokens: [wordnet_lemmatizer.lemmatize(word) for word in tokens])
df["lemmitized_str"]=df["lemmatized"].apply(lambda x: ' '.join([word for word in x]))
features= None
vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(df["lemmitized_str"])
X_train, X_test, y_train, y_test = train_test_split(
    features, df["spam"], test_size=0.2, random_state=42
)
model = train_decision_tree(X_train, y_train)
y_pred = predict_decision_tree(model, X_test)
accuracy = evaluate_decision_tree(model, X_train, y_train, X_test, y_test)
conf_matrix=get_confusion_matrix(y_test, y_pred)
print("Decision Tree Matrix:",conf_matrix)
print("Decision Tree Accuracy:", accuracy)
model2= train_knn(X_train, y_train, n_neighbors=5)
y_pred2 = predict_knn(model2, X_test)
accuracy2 = evaluate_knn(model2, X_train, y_train, X_test, y_test)
conf_matrix2=get_confusion_matrix(y_test, y_pred2)
print("KNN Matrix:",conf_matrix2)
print("KNN Accuracy:", accuracy2)
model3= train_svm(X_train, y_train)
y_pred3 = predict_svm(model3, X_test)
accuracy3 = evaluate_svm(model3,  X_train, y_train, X_test, y_test)
conf_matrix3=get_confusion_matrix(y_test, y_pred3)
print("SVM Matrix:",conf_matrix3)
print("SVM Accuracy:", accuracy3)
model4= train_naive_bayes(X_train.toarray(), y_train)
y_pred4 = predict_naive_bayes(model4, X_test.toarray())
accuracy4 = evaluate_naive_bayes(model4, X_train.toarray(), y_train, X_test.toarray(), y_test)
conf_matrix4=get_confusion_matrix(y_test, y_pred4)
print("Naive Bayes Matrix:",conf_matrix4)
print("Naive Bayes Accuracy:", accuracy4)
model5=train_logistic_regression(X_train, y_train)
y_pred5 = predict_logistic_regression(model5, X_test)
accuracy5 = evaluate_logistic_regression(model5,  X_train, y_train, X_test, y_test)
conf_matrix5=get_confusion_matrix(y_test, y_pred5)
print("Logistic Regression Matrix:",conf_matrix5)
print("Logistic Regression Accuracy:", accuracy5)

# ============== اضافة الدوك =======

print("--- Training ANN for Spam ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
model_ann = train_ann_email(X_train_dense, y_train)
accuracy_ann = evaluate_ann(model_ann, X_test_dense, y_test)
print("ANN Accuracy:", accuracy_ann)
y_pred_prob = model_ann.predict(X_test_dense)
y_pred_ann = (y_pred_prob > 0.5).astype(int)
conf_matrix_ann = get_confusion_matrix(y_test, y_pred_ann)
print("Spam ANN Matrix:\n", conf_matrix_ann)



import json

results_spam = {
    "Spam": {
        "Decision Tree": accuracy[1],
        "KNN": accuracy2[1],
        "SVM": accuracy3[1],
        "Naive Bayes": accuracy4[1],
        "Logistic Regression": accuracy5[1],
        "ANN":accuracy_ann
    }
}
CONFS_EMAILS={
    "Spam": {
        "Decision Tree": conf_matrix.tolist(),
        "KNN": conf_matrix2.tolist(),
        "SVM": conf_matrix3.tolist(),
        "Naive Bayes": conf_matrix4.tolist(),
        "Logistic Regression": conf_matrix5.tolist(),
        "ANN":conf_matrix_ann.tolist()
    }
}
with open("confs_spam.json", "w") as f:
    json.dump(CONFS_EMAILS, f)

with open("results_spam.json", "w") as f:
    json.dump(results_spam, f)