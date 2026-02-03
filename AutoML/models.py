import pickle
import re
import string
import numpy as np
from collections import Counter
#from nltk.corpus import stopwords
#
## Ensure stopwords are downloaded
#try:
#    stopword = set(stopwords.words('english'))
#except:
#    import nltk
#    nltk.download('stopwords')
#    stopword = set(stopwords.words('english'))
#
#punc = string.punctuation

##################################################
# SCRATCH IMPLEMENTATIONS (ALGORITHMS)
##################################################

class BaseRegression:
    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / (v + 1e-9)) # Epsilon to avoid division by zero

class BaseClassification:
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)

# 1. Linear Regression
class LinearRegressionScratch(BaseRegression):
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            # Gradient Descent
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 2. Logistic Regression
class LogisticRegressionScratch(BaseClassification):
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            # Gradient Descent
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])

# 3. Naive Bayes (Gaussian)
class NaiveBayesScratch(BaseClassification):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(prior + posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# 4. KNN
class KNNBase:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _get_neighbors(self, x):
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return k_nearest_labels

class KNNClassifierScratch(KNNBase, BaseClassification):
    def predict(self, X):
        y_pred = []
        for x in X:
            neighbors = self._get_neighbors(x)
            most_common = Counter(neighbors).most_common(1)[0][0]
            y_pred.append(most_common)
        return np.array(y_pred)

class KNNRegressorScratch(KNNBase, BaseRegression):
    def predict(self, X):
        y_pred = []
        for x in X:
            neighbors = self._get_neighbors(x)
            y_pred.append(np.mean(neighbors))
        return np.array(y_pred)

# 5. SVM
class SVMScratch(BaseClassification):
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.where(np.sign(approx) == -1, 0, 1)

# 6. Decision Tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeBase:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Check stopping criteria (FIXED: Added check if max_depth is not None)
        if ((self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0: 
             return Node(value=self._calculate_leaf_value(y))

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._calculate_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class DecisionTreeClassifierScratch(DecisionTreeBase, BaseClassification):
    def _calculate_leaf_value(self, y):
        return Counter(y).most_common(1)[0][0]

    def _calculate_gain(self, y, X_column, threshold):
        def entropy(y):
            hist = np.bincount(y)
            ps = hist / len(y)
            return -np.sum([p * np.log(p) for p in ps if p > 0])
            
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        return parent_entropy - child_entropy

class DecisionTreeRegressorScratch(DecisionTreeBase, BaseRegression):
    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def _calculate_gain(self, y, X_column, threshold):
        def variance(y):
            return np.var(y)
            
        parent_var = variance(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        v_l, v_r = variance(y[left_idxs]), variance(y[right_idxs])
        child_var = (n_l/n) * v_l + (n_r/n) * v_r
        return parent_var - child_var

# 7. Random Forest
class RandomForestBase:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = self._get_tree_instance()
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._aggregate_prediction(preds) for preds in tree_preds]
        return np.array(y_pred)

class RandomForestClassifierScratch(RandomForestBase, BaseClassification):
    def _get_tree_instance(self):
        return DecisionTreeClassifierScratch(max_depth=self.max_depth, 
                                             min_samples_split=self.min_samples_split, 
                                             n_features=self.n_features)
    
    def _aggregate_prediction(self, preds):
        return Counter(preds).most_common(1)[0][0]

class RandomForestRegressorScratch(RandomForestBase, BaseRegression):
    def _get_tree_instance(self):
        return DecisionTreeRegressorScratch(max_depth=self.max_depth, 
                                            min_samples_split=self.min_samples_split, 
                                            n_features=self.n_features)

    def _aggregate_prediction(self, preds):
        return np.mean(preds)

# 8. SVR
class SVRScratch(BaseRegression):
    def __init__(self, lr=0.001, epsilon=0.1, C=1.0, n_iters=1000):
        self.lr = lr
        self.epsilon = epsilon
        self.C = C
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y - y_pred
            
            dw = np.zeros(n_features)
            db = 0
            
            for i in range(n_samples):
                if np.abs(error[i]) >= self.epsilon:
                    sign = 1 if error[i] > 0 else -1
                    dw -= self.C * sign * X[i]
                    db -= self.C * sign
            
            dw += self.weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

##################################################
# USER API WRAPPERS
##################################################

def train_linear_regression(X_train, y_train):
    model = LinearRegressionScratch(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    return model

def predict_linear_regression(model, X):
    return model.predict(X)

def train_naive_bayes(X_train, y_train):
    model = NaiveBayesScratch()
    model.fit(X_train, y_train)
    return model

def predict_naive_bayes(model, X):
    return model.predict(X)

def train_decision_tree(X_train, y_train, max_depth=None):
    model = DecisionTreeClassifierScratch(max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def predict_decision_tree(model, X):
    return model.predict(X)

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNNClassifierScratch(k=n_neighbors)
    model.fit(X_train, y_train)
    return model

def predict_knn(model, X):
    return model.predict(X)

def train_svm(X_train, y_train, C=1.0, kernel='rbf'):
    lambda_param = 1.0 / (C * 100) if C > 0 else 0.01
    model = SVMScratch(lambda_param=lambda_param, n_iters=1000)
    model.fit(X_train, y_train)
    return model

def predict_svm(model, X):
    return model.predict(X)

def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    model = LogisticRegressionScratch(lr=0.01, n_iters=max_iter)
    model.fit(X_train, y_train)
    return model

def predict_logistic_regression(model, X):
    return model.predict(X)

def train_random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestClassifierScratch(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def predict_random_forest_classifier(model, X):
    return model.predict(X)

def train_decision_tree_regressor(X_train, y_train, max_depth=None):
    model = DecisionTreeRegressorScratch(max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def predict_decision_tree_regressor(model, X):
    return model.predict(X)

def train_svr(X_train, y_train, C=1.0, kernel='rbf'):
    model = SVRScratch(C=C, n_iters=1000)
    model.fit(X_train, y_train)
    return model

def predict_svr(model, X):
    return model.predict(X)

def train_random_forest_regressor(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestRegressorScratch(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def predict_random_forest_regressor(model, X):
    return model.predict(X)

def train_knn_regressor(X_train, y_train, n_neighbors=5):
    model = KNNRegressorScratch(k=n_neighbors)
    model.fit(X_train, y_train)
    return model

def predict_knn_regressor(model, X):
    return model.predict(X)

def get_confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    class_map = {cls: i for i, cls in enumerate(unique_classes)}
    for t, p in zip(y_true, y_pred):
        matrix[class_map[t], class_map[p]] += 1
    return matrix

###########################
##############################################
######  ANN For Emails Classification ########
##############################################

def train_ann_email(X_train, y_train, epochs=20, batch_size=16):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        validation_split=0.2, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stop],
                        verbose=1)
    return model

###################################################

###################################################
####### ANN For Documentation Classification ######
###################################################

def train_ann_multiclass(X_train, y_train, num_classes, epochs=50, batch_size=16): # زودنا epochs وقللنا batch
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        validation_split=0.2, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stop],
                        verbose=1)
    return model

###################################################

###################################################
############# ANN for Regression ##################
###################################################
def train_ann_regression(X_train, y_train, epochs=50, batch_size=32):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear') # Linear for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                        validation_split=0.2, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stop],
                        verbose=1)
    return model

def evaluate_ann(model, X_test, y_test, task_type='classification'):
    if task_type == 'classification':
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return accuracy
    elif task_type == 'regression':
        y_pred = model.predict(X_test, verbose=0)
        score = r2_score(y_test, y_pred)
        return score



def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', punc))
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stopword and len(word) > 1])
    return text.strip()

def convert_price_columns(df, columns):
    for col in columns:
        df[col] = df[col].str.replace(",", "").astype(float)
    return df

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

        