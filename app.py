import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# --- Page Configuration ---
st.set_page_config(
    page_title='Bank Churn Predictor',
    page_icon="🏦",
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'About': "Bank Customer Churn Prediction using Multiple ML Models"
    }
)

# --- Dark Theme CSS with Modern Aesthetics ---
st.markdown("""
    <style>
    /* Main Background - Dark gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #6c63ff;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h1 {
        color: #ffffff !important;
        font-size: 3rem !important;
    }
    
    /* Keep emojis visible */
    h1::first-letter {
        background: none !important;
        -webkit-text-fill-color: initial !important;
    }
    
    /* Text Elements */
    p, span, label, div {
        color: #e0e0e0 !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #6c63ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8b8d1 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #4ecca3 !important;
    }
    
    /* Cards/Containers */
    .card {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
        border: 1px solid rgba(108, 99, 255, 0.2);
        backdrop-filter: blur(10px);
        margin: 16px 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6c63ff 0%, #5a52d5 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.5);
        background: linear-gradient(90deg, #7c73ff 0%, #6a62e5 100%);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 30, 46, 0.6);
        border-radius: 12px;
        border: 2px dashed rgba(108, 99, 255, 0.4);
        padding: 20px;
    }
    
    [data-testid="stFileUploader"] label {
        color: #b8b8d1 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Divider */
    hr {
        border-color: rgba(108, 99, 255, 0.3);
        margin: 2rem 0;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stNumberInput, .stSlider {
        color: #e0e0e0;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(30, 30, 46, 0.8);
        color: #e0e0e0;
        border-radius: 8px;
    }
    
    input {
        background-color: rgba(30, 30, 46, 0.8) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(108, 99, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #b8b8d1 !important;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 12px;
        border-left: 4px solid #4ecca3;
        color: #e0e0e0 !important;
    }
    
    .stSuccess * {
        color: #e0e0e0 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #4ecca3 0%, #3fb88f 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #5edcb3 0%, #4fc89f 100%);
        transform: translateY(-2px);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(108, 99, 255, 0.3);
    }
    
    .stSlider > div > div > div > div {
        background: #6c63ff;
    }
    
    /* Info box custom styling */
    .info-box {
        background: rgba(108, 99, 255, 0.1);
        border-left: 4px solid #6c63ff;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
        color: #e0e0e0;
    }
    
    /* Feature importance styling */
    .feature-card {
        background: rgba(30, 30, 46, 0.6);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(108, 99, 255, 0.2);
        margin: 10px 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6c63ff;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7c73ff;
    }
    
    /* Model selector highlight */
    .stSelectbox[data-baseweb="select"] {
        background: rgba(108, 99, 255, 0.1);
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Model Performance Data ---
MODEL_PERFORMANCE = {
    'LightGBM': {
        'accuracy': 91.20,
        'precision': 93.47,
        'recall': 88.67,
        'f1': 91.02,
        'description': '🏆 Best overall performance with balanced metrics',
        'color': '#6c63ff'
    },
    'XGBoost': {
        'accuracy': 90.63,
        'precision': 92.91,
        'recall': 88.07,
        'f1': 90.42,
        'description': '🥈 Strong gradient boosting with excellent precision',
        'color': '#ff6b6b'
    },
    'Random Forest': {
        'accuracy': 86.42,
        'precision': 86.73,
        'recall': 86.13,
        'f1': 86.43,
        'description': '🌲 Robust ensemble method with good stability',
        'color': '#4ecca3'
    },
    'Decision Tree': {
        'accuracy': 86.94,
        'precision': 86.47,
        'recall': 87.72,
        'f1': 87.09,
        'description': '🌳 Simple and interpretable single tree',
        'color': '#ffd93d'
    },
    'K-Nearest Neighbors': {
        'accuracy': 71.16,
        'precision': 67.67,
        'recall': 81.52,
        'f1': 73.95,
        'description': '🎯 Distance-based classifier with high recall',
        'color': '#ff9ff3'
    },
    'Logistic Regression': {
        'accuracy': 71.29,
        'precision': 70.98,
        'recall': 72.45,
        'f1': 71.71,
        'description': '📊 Linear baseline model',
        'color': '#54a0ff'
    }
}

# --- Utility Functions ---
@st.cache_resource
def load_model(model_name):
    """Load the selected model"""
    model_files = {
        'LightGBM': 'models/bank_churn_model.pkl',
        'XGBoost': 'models/bank_churn_xgboost.pkl',
        'Random Forest': 'models/bank_churn_rf.pkl',
        'Decision Tree': 'models/bank_churn_dt.pkl',
        'K-Nearest Neighbors': 'models/bank_churn_knn.pkl',
        'Logistic Regression': 'models/bank_churn_lr.pkl'
    }
    
    model_path = model_files.get(model_name, 'models/bank_churn_model.pkl')
    
    try:
        model = joblib.load(model_path)
        return model
    except:
        try:
            model = joblib.load('models/bank_churn_model.pkl')
            st.warning(f"⚠️ {model_name} model file not found. Using LightGBM as default.")
            return model
        except Exception as e:
            st.error(f"❌ Could not load model: {e}")
            return None

# Update the load_scaler function:
@st.cache_resource
def load_scaler(scaler_path='models/bank_churn_scaler.pkl'):
    """Load scaler with error handling"""
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            sc = joblib.load(scaler_path)
            if hasattr(sc, 'transform') and hasattr(sc, 'fit'):
                scaler = sc
            else:
                st.warning("⚠️ Loaded scaler is not valid. Proceeding without scaling.")
        except Exception as e:
            st.warning(f"⚠️ Could not load scaler: {e}")
    return scaler

def preprocess_input(df, scaler=None):
    """Preprocess input data for prediction"""
    df = df.copy()
    
    # Drop id if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)
    
    # Ensure all training columns exist
    training_cols = ['id', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                     'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 
                     'Geography_Spain', 'Gender_Male']
    
    for c in training_cols:
        if c not in df.columns:
            df[c] = 0
    
    df = df[training_cols]
    
    # Convert boolean to int
    df = df.astype({col: 'int64' for col in df.select_dtypes('bool').columns})
    
    # Apply scaling if available
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    if scaler is not None and hasattr(scaler, 'transform'):
        try:
            df[num_cols] = scaler.transform(df[num_cols])
        except Exception as e:
            st.warning(f"⚠️ Could not scale features: {e}")
    
    return df

# --- Load Scaler ---
scaler = load_scaler()

# --- Header Section ---
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3.5rem; margin: 0; display: flex; align-items: center; justify-content: center; gap: 15px;'>
            <span style='font-size: 3.5rem;'>🏦</span>
            <span>Bank Churn Predictor</span>
            <span style='font-size: 3.5rem;'>🏦</span>
        </h1>
        <p style='color: #b8b8d1; font-size: 1.2rem; margin-top: 10px;'>
            Predict customer churn probability using advanced ML analytics
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Model Selector ---
st.markdown("<h2 style='text-align: center; color: #6c63ff;'>🤖 Select ML Model</h2>", unsafe_allow_html=True)

col_model1, col_model2, col_model3 = st.columns([1, 2, 1])

with col_model2:
    selected_model = st.selectbox(
        '🎯 Choose a model to use for predictions:',
        list(MODEL_PERFORMANCE.keys()),
        index=0,
        help='Select different models to compare their performance'
    )
    
    # Display model description
    model_info = MODEL_PERFORMANCE[selected_model]
    st.markdown(f"""
        <div style='background: rgba(30, 30, 46, 0.8); padding: 15px; border-radius: 12px; 
                    border-left: 4px solid {model_info['color']}; margin: 10px 0;'>
            <p style='margin: 0; color: #e0e0e0;'>{model_info['description']}</p>
        </div>
    """, unsafe_allow_html=True)

# Load selected model
with st.spinner(f'🔄 Loading {selected_model} model...'):
    model = load_model(selected_model)

if model is None:
    st.error("❌ Failed to load model. Please ensure model files exist.")
    st.stop()

# --- KPI Metrics for Selected Model ---
st.markdown(f"<h2 style='text-align: center; color: {model_info['color']};'>📊 {selected_model} Performance</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('🤖 Model', selected_model, delta='Selected')
with col2:
    st.metric('🎯 Accuracy', f"{model_info['accuracy']:.2f}%", delta=f"±{abs(90.94 - model_info['accuracy']):.2f}%")
with col3:
    st.metric('📈 Precision', f"{model_info['precision']:.2f}%", delta=f"±{abs(92.92 - model_info['precision']):.2f}%")
with col4:
    st.metric('🔄 Recall', f"{model_info['recall']:.2f}%", delta=f"±{abs(88.64 - model_info['recall']):.2f}%")

st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.markdown("<h2 style='text-align: center; color: #6c63ff;'>🎛️ Customer Profile</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("### 💳 Financial Information")
    credit = st.slider('Credit Score', 300, 900, 650, help="Customer's credit score")
    balance = st.number_input('Account Balance', min_value=0.0, value=0.0, step=100.0, format="%.2f")
    salary = st.number_input('Estimated Salary', min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    
    st.markdown("### 👤 Personal Details")
    age = st.slider('Age', 18, 100, 38)
    gender = st.selectbox('Gender', ['Female', 'Male'])
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    
    st.markdown("### 🏦 Banking Details")
    tenure = st.slider('Tenure (years)', 0, 10, 5)
    products = st.selectbox('Number of Products', [1, 2, 3, 4], index=1)
    has_card = st.radio('Has Credit Card', ['No', 'Yes'])
    active = st.radio('Is Active Member', ['No', 'Yes'])

# Create user dataframe
user_df = pd.DataFrame([{
    'CreditScore': credit,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products,
    'HasCrCard': 1 if has_card == 'Yes' else 0,
    'IsActiveMember': 1 if active == 'Yes' else 0,
    'EstimatedSalary': salary,
    'Geography': geography,
    'Gender': gender
}])

# --- Main Content Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Single Prediction", "📦 Batch Prediction", "📊 Model Comparison", "🧠 Model Insights"])

# --- TAB 1: Single Prediction ---
with tab1:
    st.markdown(f"<h2 style='color: {model_info['color']};'>🔮 Single Customer Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown("### 📋 Customer Information")
    st.dataframe(user_df.style.background_gradient(cmap='viridis'), use_container_width=True)
    
    st.markdown("---")
    
    if st.button('🚀 Predict Churn Probability', use_container_width=True):
        with st.spinner('🔄 Analyzing customer data...'):
            try:
                proc = preprocess_input(user_df, scaler)
                prob = model.predict_proba(proc)[:, 1][0]
                pred = int(prob >= 0.5)
                
                # Display result with color coding
                st.markdown(f"### 🎯 Prediction Result ({selected_model})")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if pred == 1:
                        st.error(f"### ⚠️ HIGH CHURN RISK")
                        st.metric('Churn Probability', f"{prob:.1%}", delta="Will Exit", delta_color="inverse")
                    else:
                        st.success(f"### ✅ LOW CHURN RISK")
                        st.metric('Churn Probability', f"{prob:.1%}", delta="Will Stay", delta_color="normal")
                
                with col_res2:
                    # Progress bar visualization
                    st.markdown("### 📊 Probability Scale")
                    st.progress(float(prob))
                    
                    if prob < 0.3:
                        st.success("🟢 Low Risk Zone")
                    elif prob < 0.7:
                        st.warning("🟡 Medium Risk Zone")
                    else:
                        st.error("🔴 High Risk Zone")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
    
    # SHAP Explanation
    st.markdown("---")
    st.markdown(f"<h2 style='color: {model_info['color']};'>🧠 Feature Impact Analysis</h2>", unsafe_allow_html=True)
    
    if st.button('📊 Generate SHAP Explanation', use_container_width=True):
        with st.spinner('🔄 Generating explanations...'):
            try:
                proc = preprocess_input(user_df, scaler)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(proc)
                
                # Create matplotlib figure with dark theme
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
                
                shap.summary_plot(shap_values, proc, plot_type="bar", show=False)
                ax.set_facecolor('#1a1a2e')
                
                st.pyplot(fig, use_container_width=True)
                plt.close()
                
                st.info(f"💡 **Interpretation**: Features pushing the prediction higher increase churn risk for {selected_model}.")
                
            except Exception as e:
                st.warning(f"⚠️ SHAP visualization not available: {e}")
                st.info("💡 SHAP plots require proper environment setup. The prediction still works correctly.")

# --- TAB 2: Batch Prediction ---
with tab2:
    st.markdown(f"<h2 style='color: {model_info['color']};'>📦 Batch Prediction from CSV</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
        <strong>📄 Upload Requirements:</strong><br>
        • CSV file with columns: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender<br>
        • Optional: id column (will be preserved in results)<br>
        • Ensure Geography is one of: France, Germany, Spain<br>
        • Ensure Gender is one of: Male, Female
        </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader('📤 Upload CSV File', type=['csv'])
    
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"📊 Loaded {len(df)} rows")
        
        if st.button('🚀 Run Batch Predictions', use_container_width=True):
            with st.spinner(f'🔄 Processing predictions with {selected_model}...'):
                try:
                    proc = preprocess_input(df.copy(), scaler)
                    preds = model.predict(proc)
                    probs = model.predict_proba(proc)[:, 1]
                    
                    df['Exited_Pred'] = preds
                    df['Exited_Prob'] = np.round(probs, 4)
                    df['Risk_Level'] = pd.cut(probs, bins=[0, 0.3, 0.7, 1.0], 
                                               labels=['Low', 'Medium', 'High'])
                    df['Model_Used'] = selected_model
                    
                    st.success(f'✅ Predictions completed successfully with {selected_model}!')
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Customers", len(df))
                    with col2:
                        st.metric("High Risk", len(df[df['Risk_Level'] == 'High']))
                    with col3:
                        churn_rate = (preds.sum() / len(preds)) * 100
                        st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                    with col4:
                        st.metric("Model Used", selected_model)
                    
                    st.markdown("### 📊 Results Preview")
                    st.dataframe(df.head(20), use_container_width=True)
                    
                    # Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=probs,
                        nbinsx=50,
                        marker_color=model_info['color'],
                        marker_line_color='white',
                        marker_line_width=1
                    ))
                    
                    fig.update_layout(
                        title=f'Churn Probability Distribution ({selected_model})',
                        xaxis_title='Probability',
                        yaxis_title='Count',
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(30,30,46,0.8)',
                        font=dict(color='#e0e0e0')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='📥 Download Predictions CSV',
                        data=csv,
                        file_name=f'churn_predictions_{selected_model.lower().replace(" ", "_")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during batch prediction: {e}")

# --- TAB 3: Model Comparison ---
with tab3:
    st.markdown("<h2 style='color: #6c63ff;'>📊 Compare All Models</h2>", unsafe_allow_html=True)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(MODEL_PERFORMANCE.keys()),
        'Accuracy': [MODEL_PERFORMANCE[m]['accuracy'] for m in MODEL_PERFORMANCE.keys()],
        'Precision': [MODEL_PERFORMANCE[m]['precision'] for m in MODEL_PERFORMANCE.keys()],
        'Recall': [MODEL_PERFORMANCE[m]['recall'] for m in MODEL_PERFORMANCE.keys()],
        'F1-Score': [MODEL_PERFORMANCE[m]['f1'] for m in MODEL_PERFORMANCE.keys()]
    })
    
    # Highlight selected model
    def highlight_selected(row):
        if row['Model'] == selected_model:
            return ['background-color: rgba(108, 99, 255, 0.3)'] * len(row)
        return [''] * len(row)
    
    st.markdown("### 📈 Performance Metrics Table")
    st.dataframe(
        comparison_df.style.apply(highlight_selected, axis=1).format({
            'Accuracy': '{:.2f}%',
            'Precision': '{:.2f}%',
            'Recall': '{:.2f}%',
            'F1-Score': '{:.2f}'
        }),
        use_container_width=True
    )
    
    st.info(f"💡 Currently selected model: **{selected_model}** (highlighted in purple)")
    
    # Comparison charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Accuracy comparison
        fig1 = go.Figure(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df['Accuracy'],
            marker_color=[MODEL_PERFORMANCE[m]['color'] for m in comparison_df['Model']],
            text=comparison_df['Accuracy'].apply(lambda x: f'{x:.2f}%'),
            textposition='outside'
        ))
        
        fig1.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,46,0.8)',
            font=dict(color='#e0e0e0'),
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_chart2:
        # F1-Score comparison
        fig2 = go.Figure(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df['F1-Score'],
            marker_color=[MODEL_PERFORMANCE[m]['color'] for m in comparison_df['Model']],
            text=comparison_df['F1-Score'].apply(lambda x: f'{x:.2f}'),
            textposition='outside'
        ))
        
        fig2.update_layout(
            title='Model F1-Score Comparison',
            xaxis_title='Model',
            yaxis_title='F1-Score',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,46,0.8)',
            font=dict(color='#e0e0e0'),
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Radar chart for all metrics
    st.markdown("### 🕸️ Multi-Metric Radar Comparison")
    
    fig_radar = go.Figure()
    
    for model_name in MODEL_PERFORMANCE.keys():
        metrics = MODEL_PERFORMANCE[model_name]
        fig_radar.add_trace(go.Scatterpolar(
            r=[metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=model_name,
            line_color=metrics['color'],
            opacity=0.7 if model_name == selected_model else 0.3
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(108, 99, 255, 0.2)'
            ),
            bgcolor='rgba(30, 30, 46, 0.8)'
        ),
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=500,
        title=f'All Models Performance Radar (Selected: {selected_model})'
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Model recommendations
    st.markdown("### 💡 Model Selection Guide")
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        st.markdown("""
            <div class='feature-card'>
                <h4 style='color: #6c63ff;'>🏆 Best Overall</h4>
                <p><strong>LightGBM</strong></p>
                <p style='font-size: 0.9rem;'>Highest accuracy and F1-score. Best for production deployment.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_rec2:
        st.markdown("""
            <div class='feature-card'>
                <h4 style='color: #ff6b6b;'>🎯 Best Precision</h4>
                <p><strong>XGBoost</strong></p>
                <p style='font-size: 0.9rem;'>Minimizes false positives. Use when false alarms are costly.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_rec3:
        st.markdown("""
            <div class='feature-card'>
                <h4 style='color: #ff9ff3;'>🔍 Best Recall</h4>
                <p><strong>KNN</strong></p>
                <p style='font-size: 0.9rem;'>Catches most churners. Use when missing churners is critical.</p>
            </div>
        """, unsafe_allow_html=True)

# --- TAB 4: Model Insights ---
with tab4:
    st.markdown(f"<h2 style='color: {model_info['color']};'>🧠 {selected_model} Feature Importance</h2>", unsafe_allow_html=True)
    
    try:
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            imp = pd.DataFrame({
                'Feature': model.feature_name_ if hasattr(model, 'feature_name_') else [f'Feature_{i}' for i in range(len(model.feature_importances_))],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Create plotly bar chart
            fig = go.Figure(go.Bar(
                x=imp['Importance'],
                y=imp['Feature'],
                orientation='h',
                marker=dict(
                    color=imp['Importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=imp['Importance'].apply(lambda x: f'{x:.4f}'),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'{selected_model} Feature Importance Analysis',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,46,0.8)',
                font=dict(color='#e0e0e0'),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature descriptions
            st.markdown("### 📝 Top Features Explained")
            
            feature_desc = {
                'Age': 'Customer age - older customers may have different churn patterns',
                'Balance': 'Account balance - indicates financial engagement with the bank',
                'NumOfProducts': 'Number of bank products - higher engagement typically means lower churn',
                'IsActiveMember': 'Active membership status - active members are less likely to churn',
                'Geography_Germany': 'German customers may have different churn behavior patterns',
                'Geography_Spain': 'Spanish customers may have different retention rates',
                'Gender_Male': 'Gender-based behavioral patterns in banking',
                'CreditScore': 'Credit score indicates overall financial health',
                'Tenure': 'Years with bank - longer tenure usually correlates with lower churn',
                'EstimatedSalary': 'Salary level may correlate with product usage and loyalty',
                'HasCrCard': 'Credit card ownership indicates deeper engagement'
            }
            
            col_feat1, col_feat2 = st.columns(2)
            
            for idx, feat in enumerate(imp['Feature'].head(8)):
                with col_feat1 if idx % 2 == 0 else col_feat2:
                    with st.expander(f"🔍 {feat}"):
                        st.write(feature_desc.get(feat, "Important feature for churn prediction"))
                        importance_val = imp[imp['Feature']==feat]['Importance'].values[0]
                        st.metric("Importance Score", f"{importance_val:.4f}")
                        
                        # Add visual importance bar
                        st.progress(float(importance_val / imp['Importance'].max()))
            
            # Additional insights based on model type
            st.markdown("---")
            st.markdown(f"### 🎯 {selected_model} Specific Insights")
            
            if selected_model == 'LightGBM':
                st.info("""
                    **LightGBM Advantages:**
                    - ⚡ Faster training speed and higher efficiency
                    - 📊 Better accuracy with large datasets
                    - 💾 Lower memory usage
                    - 🎯 Handles categorical features natively
                """)
            elif selected_model == 'XGBoost':
                st.info("""
                    **XGBoost Advantages:**
                    - 🏆 Industry-standard gradient boosting
                    - 🛡️ Built-in regularization prevents overfitting
                    - 🔧 Highly configurable hyperparameters
                    - 📈 Excellent for structured data
                """)
            elif selected_model == 'Random Forest':
                st.info("""
                    **Random Forest Advantages:**
                    - 🌲 Robust to outliers and noise
                    - 🎲 Low risk of overfitting
                    - 🔍 Good for feature selection
                    - 📊 Works well with imbalanced data
                """)
            elif selected_model == 'Decision Tree':
                st.info("""
                    **Decision Tree Advantages:**
                    - 🌳 Highly interpretable and explainable
                    - 🚀 Fast prediction time
                    - 📋 No feature scaling required
                    - 🎯 Clear decision rules
                """)
            elif selected_model == 'K-Nearest Neighbors':
                st.info("""
                    **KNN Advantages:**
                    - 🎯 Simple and intuitive algorithm
                    - 🔄 No training phase required
                    - 📊 Naturally handles multi-class problems
                    - 🔍 High recall for minority class
                """)
            elif selected_model == 'Logistic Regression':
                st.info("""
                    **Logistic Regression Advantages:**
                    - 📊 Simple baseline model
                    - ⚡ Very fast training and prediction
                    - 🔍 Provides probability scores
                    - 💡 Easy to interpret coefficients
                """)
        
        else:
            st.warning(f"⚠️ {selected_model} does not provide feature importance scores.")
            st.info("💡 Tree-based models (LightGBM, XGBoost, Random Forest, Decision Tree) provide the best feature importance insights.")
        
    except Exception as e:
        st.error(f"❌ Could not fetch feature importance: {e}")

# --- Footer ---
# --- Footer ---
st.markdown("---")

st.markdown(
    "<div style='text-align: center; color: #b8b8d1; padding: 30px 20px;'>"
    "<div style='margin-bottom: 25px;'>"
    "<p style='font-size: 1.1rem; color: #e0e0e0;'><strong>🏦 Bank Churn Predictor</strong> | Powered by Multiple ML Models | Built with Streamlit</p>"
    "<p style='font-size: 0.9rem; color: #888;'>💡 Tip: Compare different models to find the best fit for your use case</p>"
    "</div>"
    "<div style='margin: 25px 0; padding: 20px; background: rgba(30, 30, 46, 0.6); border-radius: 12px;'>"
    "<p style='font-size: 1rem; color: #6c63ff; margin-bottom: 15px;'><strong>Connect with me:</strong></p>"
    "<p>"
    "<a href='https://www.linkedin.com/in/rohantodkar0705?originalSubdomain=in' target='_blank' style='color: #0077b5; text-decoration: none; margin: 0 15px; font-weight: 600;'>🔗 LinkedIn</a>"
    "<span style='color: #666;'>|</span>"
    "<a href='https://github.com/Rohan-Todkar-2003' target='_blank' style='color: #6c63ff; text-decoration: none; margin: 0 15px; font-weight: 600;'>💻 GitHub</a>"
    "</p>"
    "</div>"
    "<p style='font-size: 0.9rem; color: #888; margin-top: 20px;'>"
    "Developed by <strong style='color: #6c63ff;'>Rohan Todkar</strong> | © 2025"
    "</p>"
    "</div>",
    unsafe_allow_html=True
)