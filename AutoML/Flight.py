import pandas as pd
from models import *
from sklearn.preprocessing import LabelEncoder
from models import train_ann_regression, evaluate_ann
from sklearn.preprocessing import StandardScaler

# Load data
df_business = pd.read_csv("business.csv")
df_economy  = pd.read_csv("economy.csv")

# Function to clean and preprocess flights dataset
def preprocess_flights(df):
    # Convert times (dep/arr) to float hours
    df["dep_time"] = pd.to_datetime(df["dep_time"], format="%H:%M")
    df["dep_time"] = df["dep_time"].dt.hour + df["dep_time"].dt.minute/60
    
    df["arr_time"] = pd.to_datetime(df["arr_time"], format="%H:%M")
    df["arr_time"] = df["arr_time"].dt.hour + df["arr_time"].dt.minute/60

    # Convert prices
    convert_price_columns(df, ["price"])

    # Date features
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["day"]   = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.day_name()

    # Clean stop column
    df["stop"] = df["stop"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Extract number of stops
    def extract_stops(val):
        if pd.isna(val):
            return None
        val = str(val).lower()
        if "non-stop" in val:
            return 0
        match = pd.Series(val).str.extract(r"(\d+)")
        if not match.isna().all(axis=None):
            return int(match.iloc[0,0])
        return None

    df["n_stops"] = df["stop"].apply(extract_stops)

    # Parse duration
    def parse_duration(val):
        if pd.isna(val):
            return None
        val = str(val).lower().strip()

        hours = 0.0
        minutes = 0.0

        if "h" in val:
            h_part = val.split("h")[0].strip()
            try:
                hours = float(h_part)   # 👈 حولناها لـ float مش int
            except:
                hours = 0.0
            val = val.split("h")[1]
        if "m" in val:
            m_part = val.split("m")[0].strip()
            try:
                minutes = float(m_part) # 👈 برضه float
            except:
                minutes = 0.0

        return hours * 60 + minutes


    df["duration_min"] = df["time_taken"].apply(parse_duration)
    df["duration_hr"] = df["duration_min"] / 60

    # Encode categorical
    le = LabelEncoder()
    df["airline"] = le.fit_transform(df["airline"])
    df["ch_code"] = le.fit_transform(df["ch_code"])
    
    df = pd.get_dummies(df, columns=["to", "from"], drop_first=True)

    # Drop unnecessary columns
    df = df.drop(columns=["date", "stop", "time_taken"])

    return df

# Apply preprocessing
df_business = preprocess_flights(df_business)
df_economy  = preprocess_flights(df_economy)

# Check results
print(df_business.info())
print(df_economy.info())
# أضيف عمود جديد يوضح نوع التذكرة
df_business["class"] = "business"
df_economy["class"]  = "economy"

# ندمجهم مع بعض
df_all = pd.concat([df_business, df_economy], ignore_index=True)
df_all["class"]=df_all["class"].replace({"business":1,"economy":0})
X=df_all.drop(columns=["price","dayofweek"])
y=df_all["price"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_linear_regression(X_train, y_train) 
y_pred = predict_linear_regression(model, X_test)
accuracy = evaluate_linear_regression(model,  X_train, y_train, X_test, y_test)
print("Linear Regression Accuracy:", accuracy)
model2= train_random_forest_regressor(X_train, y_train)
y_pred2 = predict_random_forest_regressor(model2, X_test)   
accuracy2 = evaluate_random_forest_regressor(model2,  X_train, y_train, X_test, y_test)
print("Random Forest Regressor Accuracy:", accuracy2)
model3= train_knn_regressor(X_train, y_train, n_neighbors=5)
y_pred3 = predict_knn_regressor(model3, X_test)
accuracy3 = evaluate_knn_regressor(model3,  X_train, y_train, X_test, y_test)
print("KNN Regressor Accuracy:", accuracy3)
#model4= train_svr(X_train, y_train)
#y_pred4 = predict_svr(model4, X_test)
#accuracy4 = evaluate_svr(model4, X_train, y_train, X_test, y_test)
##print(df_all.head())
##print(df_all["class"].value_counts())


# ================= اضافة الدوك ================


print("--- Training ANN for Flight Prices ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_ann = train_ann_regression(X_train_scaled, y_train)
r2_ann = evaluate_ann(model_ann, X_test_scaled, y_test, task_type='regression')
print("ANN R2 Score:", r2_ann)


import json

results = {
    "Price Prediction": {
        "Linear Regression": accuracy[1],
        "KNN": accuracy3[1],
        "Random Forest": accuracy2[1],
        "ANN":r2_ann
    }
}

with open("results.json", "w") as f:
    json.dump(results, f)