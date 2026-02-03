import os
import uuid
import joblib
import io
import base64
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, 
    OneHotEncoder, OrdinalEncoder, FunctionTransformer, PowerTransformer, QuantileTransformer,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

import models

app = Flask(__name__, template_folder='.')
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# --- Helpers ---
def load_user_data(session_id):
    file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.csv")
    if os.path.exists(file_path): return pd.read_csv(file_path)
    return None

def plot_to_base64(plt_obj):
    img = io.BytesIO()
    plt_obj.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def get_df_info(df):
    columns_info = []
    for col in df.columns:
        series = df[col]
        is_num = np.issubdtype(series.dtype, np.number)
        null_count = int(series.isnull().sum())
        unique = int(series.nunique())
        
        defaults = {}
        if null_count > 0:
            defaults['impute'] = 'median' if is_num else 'most_frequent'
        else:
            defaults['impute'] = 'none'

        if is_num:
            defaults['scale'] = 'robust' if abs(series.skew()) > 2 else 'standard'
            defaults['transform'] = 'power' if abs(series.skew()) > 1 else 'none'
        else:
            defaults['encode'] = 'onehot' if unique < 15 else 'ordinal'

        columns_info.append({
            "name": col,
            "type": "Numeric" if is_num else "Categorical",
            "nulls": null_count,
            "unique": unique,
            "defaults": defaults
        })
    return columns_info

def get_all_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    def load_json(filename):
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f: return json.load(f)
        return {}

    spam_res = load_json("results_spam.json")
    price_res = load_json("results.json")
    doc_res = load_json("results_DOC.json")
    intel_res_custom = load_json("results_intel.json")
    intel_res_resnet = load_json("results_intel (2).json")
    
    doc_conf = load_json("results_DOC_conf.json")
    intel_conf_custom = load_json("confs_intel.json")
    intel_conf_resnet = load_json("confs_intel (2).json")
    spam_conf = load_json("confs_spam.json")

    data = []

    def add_entry(proj, algo, acc=None, cm=None):
        for entry in data:
            if entry['Project'] == proj and entry['Algorithm'] == algo:
                if acc is not None: entry['Accuracy'] = acc
                if cm is not None: entry['ConfusionMatrix'] = cm
                return
        data.append({
            'Project': proj, 
            'Algorithm': algo, 
            'Accuracy': acc if acc is not None else 0,
            'ConfusionMatrix': cm
        })

    if "Spam" in spam_res:
        for alg, acc in spam_res["Spam"].items(): add_entry('Spam Detection', alg, acc=acc)
    if "Spam" in spam_conf:
        for alg, cm in spam_conf["Spam"].items(): add_entry('Spam Detection', alg, cm=cm)

    if "Price Prediction" in price_res:
        for alg, acc in price_res["Price Prediction"].items(): add_entry('Price Prediction', alg, acc=acc)

    if "DOC_Class" in doc_res:
        for alg, acc in doc_res["DOC_Class"].items(): add_entry('Doc Classification', alg, acc=acc)
    if "DOC_Class" in doc_conf:
        for alg, cm in doc_conf["DOC_Class"].items(): add_entry('Doc Classification', alg, cm=cm)

    target_intel = "Intel Image Classification"
    if "Intel_Image" in intel_res_custom:
        for alg, acc in intel_res_custom["Intel_Image"].items(): add_entry(target_intel, alg, acc=acc)
    if "Intel_Image" in intel_conf_custom:
        for alg, cm in intel_conf_custom["Intel_Image"].items(): add_entry(target_intel, alg, cm=cm)
            
    if "Intel Image Classification" in intel_res_resnet:
        for alg, acc in intel_res_resnet["Intel Image Classification"].items(): add_entry(target_intel, alg, acc=acc)
    if "Intel Image Classification" in intel_conf_resnet:
        for alg, cm in intel_conf_resnet["Intel Image Classification"].items(): add_entry(target_intel, alg, cm=cm)

    return data

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def show_dashboard():
    data = get_all_results()
    projects = sorted(list(set([d['Project'] for d in data])))
    project_models = {p: [] for p in projects}
    for d in data:
        project_models[d['Project']].append(d['Algorithm'])
    for p in project_models:
        project_models[p] = sorted(list(set(project_models[p])))

    accuracy_plot = None
    if data:
        try:
            df_res = pd.DataFrame(data)
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=df_res, x='Project', y='Accuracy', hue='Algorithm', palette='viridis')
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
            plt.title('Global Model Accuracy Comparison', fontsize=14, fontweight='bold')
            plt.ylim(0, 1.15)
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            accuracy_plot = plot_to_base64(plt)
            plt.close()
        except Exception as e:
            print(f"Error plotting: {e}")
            plt.close()

    return render_template('dashboard.html', plot=accuracy_plot, table_data=data, projects=projects, project_models=project_models)

@app.route('/filter_dashboard', methods=['POST'])
def filter_dashboard():
    filters = request.json
    selected_project = filters.get('project')
    selected_algos = filters.get('algorithms')
    
    data = get_all_results()
    filtered_data = []
    
    for row in data:
        match_proj = (selected_project == 'All') or (row['Project'] == selected_project)
        match_algo = (selected_algos == 'All') or (row['Algorithm'] in selected_algos)
        if match_proj and match_algo:
            filtered_data.append(row)
            
    acc_plot_b64 = None
    if filtered_data:
        try:
            df_res = pd.DataFrame(filtered_data)
            plt.figure(figsize=(10, 6))
            x_axis = 'Project' if len(set(df_res['Project'])) > 1 else 'Algorithm'
            hue = 'Algorithm' if x_axis == 'Project' else None
            ax = sns.barplot(data=df_res, x=x_axis, y='Accuracy', hue=hue, palette='viridis')
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
            plt.title(f'Performance: {selected_project}', fontsize=14, fontweight='bold')
            plt.ylim(0, 1.15)
            if hue: plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            acc_plot_b64 = plot_to_base64(plt)
            plt.close()
        except Exception as e:
            print(f"Error plotting acc: {e}")
            plt.close()

    cm_plots = []
    for row in filtered_data:
        if row.get('ConfusionMatrix'):
            try:
                plt.figure(figsize=(5, 4))
                cm_data = row['ConfusionMatrix']
                sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f"{row['Algorithm']} CM", fontsize=10)
                plt.tight_layout()
                b64 = plot_to_base64(plt)
                cm_plots.append({'title': row['Algorithm'], 'img': b64})
                plt.close()
            except Exception as e:
                print(f"Error plotting CM: {e}")
                plt.close()

    return jsonify({'plot': acc_plot_b64, 'heatmaps': cm_plots, 'data': filtered_data})

# --- Feature Engineering & Upload ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.csv")
    try:
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
        df.to_csv(file_path, index=False)
        
        numeric_df = df.select_dtypes(include=[np.number])
        heatmap_b64 = None
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            heatmap_b64 = plot_to_base64(plt)
            plt.close()
            
        columns_info = get_df_info(df)
            
        return jsonify({"session_id": session_id, "columns": columns_info, "heatmap": heatmap_b64, "total_rows": len(df), "preview": df.head(5).replace({np.nan: None}).to_dict(orient='records')})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/engineer', methods=['POST'])
def engineer_features():
    data = request.json
    session_id = data.get('session_id')
    action = data.get('action') 
    
    df = load_user_data(session_id)
    if df is None: return jsonify({"error": "Session expired"}), 404
    
    try:
        if action == 'delete':
            col = data.get('col')
            if col in df.columns:
                df = df.drop(columns=[col])
        
        elif action == 'create':
            new_col = data.get('new_col_name')
            col1 = data.get('col1')
            col2 = data.get('col2')
            operator = data.get('operator')

            if col1 in df.columns and col2 in df.columns:
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    if operator == '+':
                        df[new_col] = df[col1] + df[col2]
                    elif operator == '-':
                        df[new_col] = df[col1] - df[col2]
                    elif operator == '*':
                        df[new_col] = df[col1] * df[col2]
                    elif operator == '/':
                        df[new_col] = df[col1] / df[col2].replace(0, np.nan)
                        df[new_col] = df[new_col].fillna(0)
                    else:
                        return jsonify({"error": "Invalid operator"}), 400
                else:
                    return jsonify({"error": "Selected columns must be numeric"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.csv")
        df.to_csv(file_path, index=False)
        
        columns_info = get_df_info(df)
        
        return jsonify({
            "status": "success",
            "columns": columns_info,
            "preview": df.head(5).replace({np.nan: None}).to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/column_details', methods=['POST'])
def column_details():
    data = request.json
    session_id = data.get('session_id')
    col_name = data.get('col_name')
    df = load_user_data(session_id)
    if df is None or col_name not in df.columns: return jsonify({"error": "Data not found"}), 404
    series = df[col_name].dropna()
    is_num = np.issubdtype(series.dtype, np.number)
    plots = {}
    stats = {}
    quality = {}
    try:
        stats['Total Rows'] = len(df)
        stats['Null Count'] = int(df[col_name].isnull().sum())
        stats['Unique Values'] = int(series.nunique())
        if is_num:
            stats['Mean'] = float(round(series.mean(), 2))
            stats['Median'] = float(round(series.median(), 2))
            stats['Std Dev'] = float(round(series.std(), 2))
            stats['Skewness'] = float(round(series.skew(), 2))
            quality['Zero Values'] = int((series == 0).sum())
            quality['Negative Values'] = int((series < 0).sum())
            plt.figure(figsize=(6, 4))
            sns.histplot(series, kde=True, color='#4F46E5')
            plots['dist'] = plot_to_base64(plt)
            plt.close()
            plt.figure(figsize=(6, 2))
            sns.boxplot(x=series, color='#F59E0B')
            plots['box'] = plot_to_base64(plt)
            plt.close()
        else:
            stats['Most Frequent'] = str(series.mode()[0]) if not series.mode().empty else "N/A"
            top_10 = series.value_counts().head(10)
            plt.figure(figsize=(6, 4))
            sns.barplot(x=top_10.values, y=top_10.index, hue=top_10.index, palette='viridis', legend=False)
            plots['dist'] = plot_to_base64(plt)
            plt.close()
            plots['box'] = None
        return jsonify({"stats": stats, "quality": quality, "plots": plots})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize_data():
    data = request.json
    session_id = data.get('session_id')
    plot_type = data.get('plot_type')
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    hue_col = data.get('hue_col')
    df = load_user_data(session_id)
    if df is None: return jsonify({"error": "Session expired"}), 404
    try:
        plt.figure(figsize=(10, 6))
        if hue_col == 'None': hue_col = None
        if y_col == 'None': y_col = None
        if plot_type == 'scatter': sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='viridis')
        elif plot_type == 'line': sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
        elif plot_type == 'bar': sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='viridis')
        elif plot_type == 'hist': sns.histplot(data=df, x=x_col, hue=hue_col, kde=True, multiple="stack")
        elif plot_type == 'box': sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='coolwarm')
        elif plot_type == 'violin': sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='muted')
        elif plot_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'{plot_type.capitalize()} Plot', fontsize=14)
        plt.tight_layout()
        plot_b64 = plot_to_base64(plt)
        plt.close()
        return jsonify({"plot": plot_b64})
    except Exception as e:
        plt.close()
        return jsonify({"error": f"Error: {str(e)}"}), 400

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    session_id = data.get('session_id')
    plan = data.get('plan')
    model_type = data.get('model')
    target_col = data.get('target')
    params = data.get('hyperparameters', {})

    df = load_user_data(session_id)
    if df is None: return jsonify({"error": "Session expired"}), 404

    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        if y.isnull().any(): y = y.fillna(y.median() if np.issubdtype(y.dtype, np.number) else y.mode()[0])
        
        # --- FIX: Handle Target (y) Encoding & Scaling ---
        is_classification = model_type.endswith('_clf') or model_type == 'log_reg' or model_type == 'nb_clf'
        
        y_scaler = None
        if is_classification:
            # If target is string/object (e.g. 'Yes', 'No'), convert to integers (0, 1)
            if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)
        else:
            # For Regression: Use scaler to avoid exploding gradients in scratch models (SVR/Linear)
            if not pd.api.types.is_numeric_dtype(y):
                return jsonify({"error": f"Target column '{target_col}' must be numeric for regression."}), 400
            
            y_scaler = StandardScaler()
            # We reshape to (N, 1) for scaler, then flatten back
            y = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

        # 1. Pipeline
        transformers = []
        for col in X.columns:
            if col not in plan: continue
            settings = plan[col]
            steps = []
            is_numeric = np.issubdtype(X[col].dtype, np.number)
            imp = settings.get('impute', 'none')
            if imp == 'mean': steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif imp == 'median': steps.append(('imputer', SimpleImputer(strategy='median')))
            elif imp == 'most_frequent': steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            elif imp == 'zero': steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
            if is_numeric:
                tr = settings.get('transform', 'none')
                if tr == 'log': steps.append(('log', FunctionTransformer(np.log1p)))
                elif tr == 'power': steps.append(('power', PowerTransformer()))
                sc = settings.get('scale', 'none')
                if sc == 'standard': steps.append(('scaler', StandardScaler()))
                elif sc == 'minmax': steps.append(('scaler', MinMaxScaler()))
                elif sc == 'robust': steps.append(('scaler', RobustScaler()))
            else:
                enc = settings.get('encode', 'none')
                if enc == 'onehot': steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                elif enc == 'ordinal': steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            if steps or is_numeric or (not is_numeric and settings.get('encode') != 'none'):
                if not steps: steps.append(('identity', FunctionTransformer(lambda x: x)))
                transformers.append((f'proc_{col}', Pipeline(steps), [col]))
        
        if not transformers: return jsonify({"error": "No valid features!"}), 400
        preprocessor = ColumnTransformer(transformers, remainder='drop')

        # Split Data (y is already encoded/scaled)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- FIX: Ensure Dense Arrays for Scratch Models ---
        X_train_trans = preprocessor.fit_transform(X_train)
        if hasattr(X_train_trans, 'toarray'): X_train_trans = X_train_trans.toarray()
        
        X_test_trans = preprocessor.transform(X_test)
        if hasattr(X_test_trans, 'toarray'): X_test_trans = X_test_trans.toarray()

        # Ensure targets are pure numpy arrays (flattened)
        if hasattr(y_train, 'values'): y_train = y_train.values.ravel()
        if hasattr(y_test, 'values'): y_test = y_test.values.ravel()

        def get_int(k, d): return int(params.get(k, d)) if params.get(k) else d
        def get_float(k, d): return float(params.get(k, d)) if params.get(k) else d
        def get_str(k, d): return str(params.get(k, d))

        model = None
        # Classification
        if model_type == 'rf_clf':
            model = models.train_random_forest_classifier(X_train_trans, y_train, n_estimators=get_int('n_estimators', 100), max_depth=get_int('max_depth', None))
        elif model_type == 'log_reg':
            model = models.train_logistic_regression(X_train_trans, y_train, C=get_float('C', 1.0), max_iter=get_int('max_iter', 1000))
        elif model_type == 'nb_clf':
            model = models.train_naive_bayes(X_train_trans, y_train)
        elif model_type == 'dt_clf':
            model = models.train_decision_tree(X_train_trans, y_train, max_depth=get_int('max_depth', None))
        elif model_type == 'knn_clf':
            model = models.train_knn(X_train_trans, y_train, n_neighbors=get_int('n_neighbors', 5))
        elif model_type == 'svm_clf':
            model = models.train_svm(X_train_trans, y_train, C=get_float('C', 1.0), kernel=get_str('kernel', 'rbf'))
        # Regression
        elif model_type == 'rf_reg':
            model = models.train_random_forest_regressor(X_train_trans, y_train, n_estimators=get_int('n_estimators', 100), max_depth=get_int('max_depth', None))
        elif model_type == 'lin_reg':
            model = models.train_linear_regression(X_train_trans, y_train)
        elif model_type == 'dt_reg':
            model = models.train_decision_tree_regressor(X_train_trans, y_train, max_depth=get_int('max_depth', None))
        elif model_type == 'knn_reg':
            model = models.train_knn_regressor(X_train_trans, y_train, n_neighbors=get_int('n_neighbors', 5))
        elif model_type == 'svr_reg':
            model = models.train_svr(X_train_trans, y_train, C=get_float('C', 1.0), kernel=get_str('kernel', 'rbf'))
        else: return jsonify({"error": "Unknown model type"}), 400

        train_preds = model.predict(X_train_trans)
        test_preds = model.predict(X_test_trans)
        
        # --- FIX: Inverse Scale if Regressor ---
        if y_scaler is not None:
            train_preds = y_scaler.inverse_transform(train_preds.reshape(-1, 1)).ravel()
            test_preds = y_scaler.inverse_transform(test_preds.reshape(-1, 1)).ravel()
            # Inverse scale true values for metric calculation
            y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
            y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

        metrics = {}
        if is_classification:
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            metrics['Train Acc'] = f"{train_acc*100:.1f}%"
            metrics['Test Acc'] = f"{test_acc*100:.1f}%"
            metrics['Status'] = "High Overfitting" if train_acc - test_acc > 0.15 else "Good Fit"
        else:
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            metrics['Train R2'] = f"{train_r2:.3f}"
            metrics['Test R2'] = f"{test_r2:.3f}"
            metrics['MAE'] = f"{mean_absolute_error(y_test, test_preds):.3f}"

        # Save Pipeline
        full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        model_filename = f"{session_id}_model.pkl"
        joblib.dump(full_pipeline, os.path.join(MODELS_FOLDER, model_filename))

        # Generate Cleaned CSV
        proc_filename = None
        try:
            X_full_trans = preprocessor.fit_transform(X)
            if hasattr(X_full_trans, 'toarray'): X_full_trans = X_full_trans.toarray()
            try: feature_names = [f.split('__')[-1] for f in preprocessor.get_feature_names_out()]
            except: feature_names = [f"f{i}" for i in range(X_full_trans.shape[1])]
            processed_df = pd.DataFrame(X_full_trans, columns=feature_names)
            processed_df[target_col] = y if isinstance(y, np.ndarray) else y.values
            proc_filename = f"{session_id}_cleaned.csv"
            processed_df.to_csv(os.path.join(MODELS_FOLDER, proc_filename), index=False)
        except Exception as e:
            print(f"Clean CSV Error: {e}")

        return jsonify({
            "status": "success", 
            "metrics": metrics, 
            "download_url": f"/download/{model_filename}",
            "processed_url": f"/download/{proc_filename}" if proc_filename else None
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_live():
    data = request.json
    session_id = data.get('session_id')
    inputs = data.get('inputs')
    try:
        model_path = os.path.join(MODELS_FOLDER, f"{session_id}_model.pkl")
        pipeline = joblib.load(model_path)
        df_orig = load_user_data(session_id)
        input_df = pd.DataFrame([inputs])
        for col in input_df.columns:
            if col in df_orig.columns:
                if np.issubdtype(df_orig[col].dtype, np.number):
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        prediction = pipeline.predict(input_df)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/generate_code', methods=['POST'])
def generate_code():
    data = request.json
    session_id = data.get('session_id')
    plan = data.get('plan')
    target = data.get('target')
    model_type = data.get('model')
    if not session_id: session_id = str(uuid.uuid4())

    model_classes = {
        'rf_clf': 'RandomForestClassifier', 'log_reg': 'LogisticRegression',
        'nb_clf': 'GaussianNB', 'dt_clf': 'DecisionTreeClassifier',
        'knn_clf': 'KNeighborsClassifier', 'svm_clf': 'SVC',
        'rf_reg': 'RandomForestRegressor', 'lin_reg': 'LinearRegression',
        'dt_reg': 'DecisionTreeRegressor', 'knn_reg': 'KNeighborsRegressor',
        'svr_reg': 'SVR'
    }
    imports = "from sklearn.ensemble import *\nfrom sklearn.linear_model import *\nfrom sklearn.naive_bayes import *\nfrom sklearn.tree import *\nfrom sklearn.neighbors import *\nfrom sklearn.svm import *"

    code = f"# Auto-Generated Pipeline\nimport pandas as pd\nimport numpy as np\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import *\nfrom sklearn.impute import *\n{imports}\nfrom sklearn.model_selection import train_test_split\n\n# Load Data\ndf = pd.read_csv('your_data.csv')\n"
    if target: code += f"X = df.drop(columns=['{target}'])\ny = df['{target}']\n"
    else: code += "X = df.iloc[:, :-1]\ny = df.iloc[:, -1]\n"
    code += "\ntransformers = []\n"
    if plan:
        for col, settings in plan.items():
            steps = []
            if settings.get('impute') != 'none': steps.append(f"SimpleImputer(strategy='{settings.get('impute')}')")
            if settings.get('scale') != 'none': steps.append(f"{settings.get('scale', 'standard').capitalize()}Scaler()")
            if settings.get('encode') == 'onehot': steps.append("OneHotEncoder(handle_unknown='ignore')")
            if steps:
                steps_str = ", ".join([f"('{i}', {s})" for i, s in enumerate(steps)])
                code += f"pipe_{col} = Pipeline([{steps_str}])\ntransformers.append(('{col}', pipe_{col}, ['{col}']))\n"

    model_class = model_classes.get(model_type, 'RandomForestClassifier')
    code += f"\npreprocessor = ColumnTransformer(transformers)\nmodel = {model_class}()\npipeline = Pipeline([('prep', preprocessor), ('model', model)])\n\n# Splitting & Training\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\npipeline.fit(X_train, y_train)\nprint('Score:', pipeline.score(X_test, y_test))"
    
    filename = f"{session_id}_script.py"
    with open(os.path.join(MODELS_FOLDER, filename), 'w') as f: f.write(code)
    return jsonify({"download_url": f"/download/{filename}"})

@app.route('/download/<filename>', methods=['GET'])
def download_model(filename):
    return send_file(os.path.join(MODELS_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)