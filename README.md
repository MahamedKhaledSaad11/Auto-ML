# 🧠 AutoML Master Class

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn)
![TailwindCSS](https://img.shields.io/badge/Tailwind-Frontend-38B2AC?style=for-the-badge&logo=tailwind-css)

> **A powerful No-Code Machine Learning platform that empowers users to build, visualize, and deploy ML pipelines without writing a single line of code.**

## 🚀 Live Demo
**[View Live Application](https://momosaad11.pythonanywhere.com/)** 

---

## 📖 Project Overview

**AutoML Master Class** bridges the gap between raw data and production-ready models. It eliminates the barrier to entry for machine learning by providing an intuitive graphical interface for the entire data science workflow.

Unlike standard black-box tools, this project emphasizes **transparency and education**—users can not only train models but also visualize their data structure and **generate the actual Python code** used to build the pipeline.

### Key Capabilities
* **📂 Intelligent Data Ingestion:** Supports CSV and Excel files with automatic type detection (Numeric vs Categorical).
* **🛠️ Interactive Feature Engineering:** Create new features, scale data (Standard/MinMax/Robust), and handle missing values via a simple UI.
* **📊 Automated EDA:** Instantly generate correlation heatmaps, distribution plots, and scatter charts to understand data patterns.
* **🤖 Multi-Model Training:** Train and compare multiple algorithms (Random Forest, SVM, Logistic Regression, etc.) in real-time.
* **📝 Code Generation Engine:** One-click export of the complete, reproducible Python script for the trained pipeline.

---

## 🛠️ Technical Architecture

This project is built as a full-stack monolithic application ensuring low latency and high data integrity.

### **Backend (Python & Flask)**
* **Core Logic:** `Flask` server handles session management and routing.
* **ML Engine:** `Scikit-Learn` pipelines are dynamically constructed based on user JSON configurations.
* **Data Processing:** `Pandas` and `Numpy` handle vectorized operations for high-speed feature engineering.
* **Visualization:** `Matplotlib` and `Seaborn` generate static plots rendered as Base64 strings for the frontend.

### **Frontend (HTML & Tailwind)**
* **UI/UX:** Responsive design built with **Tailwind CSS**.
* **Interactivity:** Vanilla JavaScript handles asynchronous API calls (AJAX) to prevent page reloads during training.
* **Icons:** Integration with FontAwesome for intuitive visual cues.

---

## 💻 Installation & Setup

Want to run this locally? Follow these steps:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/MahamedKhaledSaad11/Auto-ML](https://github.com/MahamedKhaledSaad11/Auto-ML)
    cd Auto-ML
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python app.py
    ```

4.  **Access the Dashboard**
    Open your browser and navigate to `http://127.0.0.1:5000`

---

## 🧬 Feature Engineering Capabilities

The platform supports a comprehensive list of transformations:

| Operation | Methods Available |
| :--- | :--- |
| **Imputation** | Mean, Median, Most Frequent, Constant |
| **Encoding** | One-Hot Encoding, Ordinal Encoding |
| **Scaling** | Standard Scaler, MinMax Scaler, Robust Scaler |
| **Transformation** | Log, Power (Yeo-Johnson), Quantile |

---

## 🛡️ License

This project is open-source and available under the **MIT License**.

---

*Developed by [Mohamed Khaled Saad, Ahmed Saif Elislam, Zeyad Tarek Ahmed, Yassmin Salah Adel, Sara Hesham Hassan, Ahmed Ayman Sayed]*
