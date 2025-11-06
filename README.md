# 🏦 Bank Customer Churn Prediction

<div align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6B6B?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-6C63FF?style=for-the-badge)

**Predict bank customer churn with 90.9% accuracy using ensemble ML models**

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)
![Model Accuracy](https://img.shields.io/badge/Accuracy-90.9%25-brightgreen?style=for-the-badge)

</div>

---

## 📑 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🚀 Key Features](#-key-features)
- [📊 Dataset Information](#-dataset-information)
- [🤖 ML Models Performance](#-ml-models-performance)
- [🛠️ Installation & Setup](#️-installation--setup)
- [💻 Usage Guide](#-usage-guide)
- [📈 Model Architecture](#-model-architecture)
- [🎮 Streamlit Web App](#-streamlit-web-app)
- [🔍 Technical Insights](#-technical-insights)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👨‍💻 Author](#-author)

---

## 🎯 Project Overview

This project implements a **comprehensive Machine Learning pipeline** to predict bank customer churn using multiple classification algorithms. The solution addresses the critical business problem of customer retention in the banking sector by identifying customers likely to leave, enabling proactive retention strategies.

### 🎪 Key Highlights

| Feature | Description |
|---------|-------------|
| **End-to-End Pipeline** | Data preprocessing → EDA → Feature engineering → Model training → Evaluation → Deployment |
| **Multi-Model Comparison** | 6 different ML algorithms with detailed performance analysis |
| **Production-Ready** | Interactive Streamlit application for real-time predictions |
| **Advanced Analytics** | SHAP explanations, feature importance, and model interpretability |
| **Class Imbalance Handling** | SMOTE implementation for balanced training |

---

## 🚀 Key Features

### 🔮 Prediction Capabilities

- **Single Customer Prediction** — Real-time churn probability for individual customers
- **Batch Processing** — CSV upload for multiple customer predictions
- **Risk Stratification** — Low/Medium/High risk categorization
- **Probability Scores** — Continuous churn likelihood (0-100%)

### 📊 Analytical Features

- **Multi-Model Comparison** — Side-by-side performance evaluation
- **Feature Importance** — SHAP-based explanation system
- **Interactive Visualizations** — Dynamic charts and performance metrics
- **Model Insights** — Detailed algorithm-specific analysis

### 🎯 Business Impact

- 90.9% prediction accuracy with LightGBM model
- Proactive retention strategies for at-risk customers
- Minimize customer acquisition costs
- Data-driven decisions for marketing teams

---

## 📊 Dataset Information

### 📦 Data Sources

| Metric | Value |
|--------|-------|
| **Training Set** | 165,034 customers × 14 features |
| **Test Set** | 110,023 customers × 13 features |
| **Target Variable** | Exited (0 = Stayed, 1 = Churned) |
| **Churn Rate** | 21.16% (34,894 customers) |

### 🏷️ Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Demographic** | Age, Gender, Geography | Customer personal information |
| **Financial** | CreditScore, Balance, EstimatedSalary | Financial health indicators |
| **Engagement** | Tenure, NumOfProducts, IsActiveMember | Banking relationship metrics |
| **Behavioral** | HasCrCard, IsActiveMember | Product usage patterns |

### 📈 Data Statistics

```
✓ No missing values in the dataset
✓ Credit Score: 656 ± 80 (mean ± std)
✓ Age Distribution: 38 ± 9 years
✓ Balance Range: $0 - $250,898
✓ Class Imbalance: 21.16% churn rate
```

---

## 🤖 ML Models Performance

### 🏆 Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | Best For |
|-------|----------|-----------|--------|----------|----------|
| **LightGBM** 🏆 | 90.94% | 92.92% | 88.64% | 90.73 | Overall Performance |
| **XGBoost** 🥈 | 90.55% | 92.73% | 88.01% | 90.31 | High Precision |
| **Random Forest** 🌲 | 86.31% | 86.76% | 85.72% | 86.24 | Stability |
| **Decision Tree** 🌳 | 86.28% | 85.76% | 87.02% | 86.38 | Interpretability |
| **K-Nearest Neighbors** 🎯 | 70.26% | 66.84% | 80.48% | 73.03 | High Recall |
| **Logistic Regression** 📊 | 70.68% | 70.76% | 70.56% | 70.66 | Baseline |

### 🎯 Key Findings

- ✅ Tree-based ensembles significantly outperform linear models
- ✅ LightGBM achieves the best balance between accuracy and F1-score
- ✅ KNN shows highest recall but lower precision
- ✅ SMOTE improved minority class prediction by ~5-7%

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- ~500MB disk space for models and data

### 📥 Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Rohan-Todkar-2003/bank-churn-prediction.git
cd bank-churn-prediction
```

#### 2. Create Virtual Environment

```bash
# On Windows
python -m venv churn_env
churn_env\Scripts\activate

# On macOS/Linux
python3 -m venv churn_env
source churn_env/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download Pre-trained Models (Optional)

```bash
# Models are included in the repository
# Or train models using the Jupyter notebook
python -m jupyter notebook notebooks/bank_churn_prediction.ipynb
```

### 📋 Requirements File

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
lightgbm==4.1.0
xgboost==1.7.6
shap==0.42.1
plotly==5.15.0
matplotlib==3.7.1
seaborn==0.12.2
joblib==1.3.2
imbalanced-learn==0.10.1
```

---

## 💻 Usage Guide

### 🚀 Running the Streamlit App

```bash
# Navigate to project directory
cd bank-churn-prediction

# Launch the web application
streamlit run app.py
```

The app will open at **http://localhost:8501** in your default browser.

### 📱 Web App Interface

#### 1. Single Prediction Tab 🔮

- Input customer details through interactive sliders and dropdowns
- Get real-time churn probability
- View SHAP feature importance explanations
- Risk level classification (Low/Medium/High)

#### 2. Batch Prediction Tab 📦

- Upload CSV files with customer data
- Process multiple customers simultaneously
- Download predictions with risk scores
- Visualize probability distributions

#### 3. Model Comparison Tab 📊

- Compare all 6 ML models side-by-side
- Interactive performance charts
- Radar plots for multi-metric comparison
- Model selection recommendations

#### 4. Model Insights Tab 🧠

- Feature importance analysis
- Algorithm-specific advantages
- Technical implementation details
- Business interpretation guidance

### 📥 Input Data Format

For batch predictions, ensure your CSV includes these columns:

```csv
CustomerId,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
15647311,619,France,Female,42,2,0.00,1,1.0,1.0,101348.88
15619304,608,Spain,Female,41,1,83807.86,1,0.0,1.0,112542.58
```

---

## 📈 Model Architecture

### 🔄 Machine Learning Pipeline

```
1. Data Loading & Inspection
   ↓
2. Exploratory Data Analysis (EDA)
   ↓
3. Data Preprocessing
   ├── Handle missing values
   ├── One-hot encoding (Geography, Gender)
   └── Feature scaling
   ↓
4. Class Balancing (SMOTE)
   ↓
5. Model Training (6 algorithms)
   ↓
6. Hyperparameter Tuning
   ↓
7. Model Evaluation & Comparison
   ↓
8. SHAP Explanation Generation
   ↓
9. Streamlit Deployment
```

### 🎯 Feature Engineering

```python
# Key preprocessing steps
- Dropped: CustomerId, Surname (non-informative)
- Encoded: Geography → Germany, Spain (one-hot)
- Encoded: Gender → Male (binary)
- Scaled: CreditScore, Age, Balance, EstimatedSalary
- Balanced: SMOTE applied (130k samples each class)
```

### ⚡ Model Training Code

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train LightGBM model
lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=7,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_resampled, y_resampled)

# Evaluate
accuracy = lgb_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
```

---

## 🎮 Streamlit Web App

### 🎨 UI/UX Features

- **Dark Theme Design** — Modern gradient background with purple accent colors
- **Interactive Cards** — Glass morphism effects with smooth animations
- **Responsive Layout** — Compatible with all screen sizes
- **Real-time Updates** — Dynamic prediction updates with visual feedback

### 🔧 Technical Implementation

```python
# App structure highlights
import streamlit as st
import joblib
import pandas as pd
from lightgbm import LGBMClassifier

# Load models with caching for performance
@st.cache_resource
def load_models():
    models = {
        'LightGBM': joblib.load('models/bank_churn_model.pkl'),
        'XGBoost': joblib.load('models/bank_churn_xgboost.pkl'),
        'Random Forest': joblib.load('models/bank_churn_rf.pkl')
    }
    return models

# Multi-page tabs for different functionalities
# Cached model loading for performance
# Real-time prediction updates
# SHAP integration for model interpretability
# CSV export with formatted results
```

---

## 🔍 Technical Insights

### 🎯 Business Impact Analysis

#### Cost-Benefit Calculation

**Assumptions:**
- Average customer lifetime value: **$5,000**
- Cost of retention campaign: **$50 per customer**
- Current churn rate: **21%**
- Model accuracy: **90.9%**

**Results:**
- Preventable revenue loss: **$4.5M per 10,000 customers**
- ROI on retention campaigns: **900%**
- Annual savings potential: **$945,000 per 10,000 customers**

### 📊 Feature Importance Insights

| Feature | Impact | Business Interpretation |
|---------|--------|------------------------|
| **Age** | 🔴 High | Older customers show different churn patterns |
| **Balance** | 🔴 High | Higher balances correlate with loyalty |
| **NumOfProducts** | 🔴 High | Multi-product users are more engaged |
| **IsActiveMember** | 🟡 Medium | Active users are less likely to churn |
| **Geography** | 🟡 Medium | Regional differences in banking behavior |
| **CreditScore** | 🟢 Low-Medium | Indirect indicator of financial stability |

### ⚡ Performance Optimization

```
✓ LightGBM for fast training & inference (50ms per prediction)
✓ Joblib for efficient model serialization
✓ Streamlit caching for improved responsiveness
✓ Batch processing for CSV predictions
✓ Minimal memory footprint design (<200MB)
```

---

## 📁 Project Structure

```
bank-churn-prediction/
│
├── 📁 models/
│   ├── bank_churn_model.pkl              # LightGBM model
│   ├── bank_churn_xgboost.pkl            # XGBoost model
│   ├── bank_churn_rf.pkl                 # Random Forest model
│   ├── bank_churn_dt.pkl                 # Decision Tree model
│   ├── bank_churn_knn.pkl                # KNN model
│   ├── bank_churn_lr.pkl                 # Logistic Regression
│   └── bank_churn_scaler.pkl             # Feature scaler
│
├── 📁 data/
│   ├── train.csv.zip                     # Training data (165K rows)
│   └── test.csv                          # Test data (110K rows)
│
├── 📁 notebooks/
│   └── bank_churn_prediction.ipynb       # Complete analysis notebook
│
├── app.py                                # Streamlit web application
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
├── .gitignore                            # Git ignore file
```

### 🗂️ File Descriptions

| File | Purpose |
|------|---------|
| `bank_churn_prediction.ipynb` | Complete ML pipeline with EDA, modeling, and evaluation |
| `app.py` | Production-ready Streamlit web application |
| `requirements.txt` | All necessary Python packages |
| `models/` | Serialized trained models for immediate use |
| `data/` | Training and testing datasets |

---

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### 🐛 Reporting Issues

- Use [GitHub Issues](https://github.com/Rohan-Todkar-2003/bank-churn-prediction/issues) to report bugs
- Include detailed descriptions and reproduction steps
- Attach relevant error messages or screenshots

### 💡 Feature Requests

- Suggest new models, visualizations, or app features
- Propose performance improvements or optimizations
- Request additional deployment platforms

### 🔧 Development Setup

```bash
# 1. Fork the repository
# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make your changes
# 4. Commit changes
git commit -m 'Add amazing feature'

# 5. Push to branch
git push origin feature/amazing-feature

# 6. Open Pull Request
```

### 📋 Contribution Areas

- ✅ Additional ML algorithms (CatBoost, SVM, Neural Networks)
- ✅ Enhanced visualization capabilities
- ✅ Database integration (PostgreSQL, MongoDB)
- ✅ REST API development (FastAPI, Flask)
- ✅ Mobile app version
- ✅ Multi-language support
- ✅ Docker containerization
- ✅ Cloud deployment (AWS, Azure, GCP)


---

## 👨‍💻 Author

### Rohan Todkar
**Machine Learning Engineer & Data Scientist**

### 🔗 Connect with Me

| Platform | Link |
|----------|------|
| **LinkedIn** | [Professional profile and experience](https://linkedin.com/in/rohan-todkar) |
| **GitHub** | [Open source projects and contributions](https://github.com/Rohan-Todkar-2003) |
| **Portfolio** | [Personal website and project showcase](https://rohans-portfolio-opal.vercel.app/) |

### 🎯 Skills & Expertise

```
Machine Learning:  Classification • Regression • Clustering • Deep Learning
Data Science:      EDA • Feature Engineering • Statistical Analysis
Tools & Tech:      Python • Scikit-learn • TensorFlow • PyTorch • SQL
```

---

## 🎯 Quick Links

| Resource | Link |
|----------|------|
| **Live Demo** | [Streamlit App](https://bank-churn-prediction.streamlit.app) |
| **Jupyter Notebook** | [View Notebook](https://github.com/Rohan-Todkar-2003/bank-churn-prediction/blob/main/notebooks/bank_churn_prediction.ipynb) |
| **Issues & Discussions** | [GitHub Issues](https://github.com/Rohan-Todkar-2003/bank-churn-prediction/issues) |

---

<div align="center">

## 🌟 Show Your Support

If this project helped you, please give it a **⭐ star** on GitHub!

**"Predicting customer behavior today, building customer loyalty tomorrow."** 🏦

---

### 📝 Disclaimer

> This project is for **educational and demonstration purposes**. Always validate models with real business data before production deployment. Not responsible for predictions made in production environments without proper testing and validation.

**Last Updated:** November 2024 | **Version:** 1.0.0

</div>
