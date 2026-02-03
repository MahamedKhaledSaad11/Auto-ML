import pandas as pd
import os
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from models import *  
from models import train_ann_multiclass, evaluate_ann
import numpy as np

main_folder = r"D:\Level 3 First Term\Machine_Learning\Machine-Learning-Algorithms-Dashboard-main\Machine-Learning-Algorithms-Dashboard-main\data\DATA"

train_data = []
test_data = []

for subfolder in os.listdir(main_folder):
    sub_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(sub_path):
        files = [f for f in os.listdir(sub_path) if f.endswith(".txt")]
        random.shuffle(files)

        train_files = files[:90]
        test_files = files[90:]

        for f in train_files:
            file_path = os.path.join(sub_path, f)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
            train_data.append([f, content, subfolder])

        
        for f in test_files:
            file_path = os.path.join(sub_path, f)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
            test_data.append([f, content, subfolder])

train_df = pd.DataFrame(train_data, columns=["filename", "content", "label"])
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.DataFrame(test_data, columns=["filename", "content", "label"])

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

train_df["clean_content"] = train_df["content"].apply(clean_text)
test_df["clean_content"] = test_df["content"].apply(clean_text)

lemmatizer = WordNetLemmatizer()

train_df["tokens"] = train_df["clean_content"].apply(lambda x: word_tokenize(x))
train_df["lemmatized"] = train_df["tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
train_df["lemmitized_str"] = train_df["lemmatized"].apply(lambda x: " ".join(x))

test_df["tokens"] = test_df["clean_content"].apply(lambda x: word_tokenize(x))
test_df["lemmatized"] = test_df["tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
test_df["lemmitized_str"] = test_df["lemmatized"].apply(lambda x: " ".join(x))

le = LabelEncoder()
train_df["label_encoded"] = le.fit_transform(train_df["label"])
test_df["label_encoded"] = le.transform(test_df["label"])

y_train = train_df["label_encoded"]
y_test = test_df["label_encoded"]

vec = TfidfVectorizer(encoding="latin-1", strip_accents="unicode", stop_words="english", max_features=3000)
X_train = vec.fit_transform(train_df["lemmitized_str"])
X_test = vec.transform(test_df["lemmitized_str"])

model = train_decision_tree(X_train, y_train)
y_pred = predict_decision_tree(model, X_test)
accuracy = evaluate_decision_tree(model,X_train, y_train, X_test, y_test)
conf_matrix=get_confusion_matrix(y_test, y_pred)
print("Doc_Class Decision Tree Matrix:",conf_matrix)
print("Doc_Class Decision Tree Accuracy:", accuracy)
model2= train_knn(X_train, y_train, n_neighbors=5)
y_pred2 = predict_knn(model2, X_test)
accuracy2 = evaluate_knn(model2,X_train, y_train, X_test, y_test)
conf_matrix2=get_confusion_matrix(y_test, y_pred2)
print("Doc_Class KNN Matrix:",conf_matrix2)
print("Doc_Class KNN Accuracy:", accuracy2)
model3= train_svm(X_train, y_train)
y_pred3 = predict_svm(model3, X_test) 
accuracy3 = evaluate_svm(model3,X_train, y_train, X_test, y_test) 
conf_matrix3=get_confusion_matrix(y_test, y_pred3)
print("Doc_Class SVM Matrix:",conf_matrix3) 
print("Doc_Class SVM Accuracy:", accuracy3)
model4= train_naive_bayes(X_train.toarray(), y_train)
y_pred4 = predict_naive_bayes(model4, X_test.toarray())
accuracy4 = evaluate_naive_bayes(model4, X_train.toarray(), y_train, X_test.toarray(), y_test)
conf_matrix4=get_confusion_matrix(y_test, y_pred4)
print("Doc_Class Naive Bayes Matrix:",conf_matrix4)
print("Doc_Class Naive Bayes Accuracy:", accuracy4)
model5=train_logistic_regression(X_train, y_train)
y_pred5 = predict_logistic_regression(model5, X_test)
accuracy5 = evaluate_logistic_regression(model5, X_train, y_train, X_test, y_test)
conf_matrix5=get_confusion_matrix(y_test, y_pred5)
print("Doc_Class Logistic Regression Matrix:",conf_matrix5)
print("Doc_Class Logistic Regression Accuracy:", accuracy5)

# ======================= اضافة الدوك =============

print("--- Training ANN for Doc Classification ---")

X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
num_classes = len(np.unique(y_train))
model_ann = train_ann_multiclass(X_train_dense, y_train, num_classes)
accuracy_ann = evaluate_ann(model_ann, X_test_dense, y_test)
y_pred_prob = model_ann.predict(X_test_dense)
y_pred_ann = np.argmax(y_pred_prob, axis=1)
conf_matrix_ann = get_confusion_matrix(y_test, y_pred_ann)
print("Doc_Class ANN Matrix:\n", conf_matrix_ann)

print("Doc_Class ANN Accuracy:", accuracy_ann)

import json

results_DOC = {
    "DOC_Class": {
        "Decision Tree": accuracy[1],
        "KNN": accuracy2[1],
        "SVM": accuracy3[1],
        "Naive Bayes": accuracy4[1],
        "Logistic Regression": accuracy5[1],
        "ANN":accuracy_ann
    }
}
CONFS_DOC={
    "DOC_Class": {
        "Decision Tree": conf_matrix.tolist(),
        "KNN": conf_matrix2.tolist(),
        "SVM": conf_matrix3.tolist(),
        "Naive Bayes": conf_matrix4.tolist(),
        "Logistic Regression": conf_matrix5.tolist(),
        "ANN":conf_matrix_ann.tolist()
    }
}
with open("results_DOC_conf.json", "w") as f:
    json.dump(CONFS_DOC, f)

with open("results_DOC.json", "w") as f:
    json.dump(results_DOC, f)