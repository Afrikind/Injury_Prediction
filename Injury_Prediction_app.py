import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import json
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE
import requests
import websocket
import threading
import time
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Custom CSS for unique styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    .stForm {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'heart_rate_data' not in st.session_state:
    st.session_state.heart_rate_data = []
if 'current_heart_rate' not in st.session_state:
    st.session_state.current_heart_rate = None
if 'smartwatch_connected' not in st.session_state:
    st.session_state.smartwatch_connected = False

# User management
def load_users():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            return json.load(f)
    return {
        'admin': {
            'password': hashlib.sha256('admin123'.encode()).hexdigest(),
            'role': 'admin'
        }
    }

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def signup_page():
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ Elite Sports Injury Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Create Your Account</h3>", unsafe_allow_html=True)
    
    users = load_users()
    
    with st.form("signup_form"):
        new_username = st.text_input("👤 Choose a username")
        new_password = st.text_input("🔒 Choose a password", type="password")
        confirm_password = st.text_input("🔐 Confirm password", type="password")
        role = st.selectbox("🎯 Select your role", ["athlete", "coach"])
        
        submitted = st.form_submit_button("🚀 Create Account")
        
        if submitted:
            if not new_username or not new_password:
                st.error("❌ Please fill in all fields")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match")
            elif new_username in users:
                st.error("❌ Username already exists")
            else:
                users[new_username] = {
                    'password': hashlib.sha256(new_password.encode()).hexdigest(),
                    'role': role
                }
                save_users(users)
                st.success("✅ Account created successfully! Please login.")
                st.session_state.show_signup = False
                st.rerun()

def login_page():
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ Elite Sports Injury Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Welcome Back!</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Login", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()
    with col2:
        if st.button("✨ Sign Up", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()
    
    if st.session_state.show_signup:
        signup_page()
        return
    
    users = load_users()
    
    with st.form("login_form"):
        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")
        
        submitted = st.form_submit_button("🚀 Login")
        
        if submitted:
            if username in users and users[username]['password'] == hashlib.sha256(password.encode()).hexdigest():
                st.session_state.authenticated = True
                st.session_state.user_role = users[username]['role']
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('injury_data (2).csv')

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data.iloc[:, :] = imputer.fit_transform(data)
    # Select features and target
    features = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries',
                'Training_Intensity', 'Recovery_Time', 'Heart_Rate']
    target = 'Likelihood_of_Injury'

    # Enhanced feature engineering with normalized values
    data['BMI'] = data['Player_Weight'] / ((data['Player_Height'] / 100) ** 2)
    
    # Normalize age to 0-1 scale (assuming age range 15-50)
    data['Age_Normalized'] = (data['Player_Age'] - 15) / 35
    
    # Training intensity based features (normalized)
    data['Training_Load'] = data['Training_Intensity'] * (1 / (data['Recovery_Time'] + 1))
    data['Training_Intensity_Ratio'] = data['Training_Intensity'] / 10  # Normalized to 0-1
    
    # Heart rate based features (normalized)
    data['Heart_Rate_Normalized'] = (data['Heart_Rate'] - 40) / 160  # Normalize to 0-1 scale
    data['Heart_Rate_Status'] = np.where(
        (data['Heart_Rate'] >= 60) & (data['Heart_Rate'] <= 100),
        0.3,  # Normal range (lower risk)
        np.where(data['Heart_Rate'] < 60, 0.6, 0.8)  # Below or above normal
    )
    
    # Recovery time features (normalized)
    data['Recovery_Quality'] = 1 / (data['Recovery_Time'] + 1)
    data['Recovery_Intensity_Ratio'] = data['Recovery_Time'] / (data['Training_Intensity'] + 1)
    
    # Combined features with balanced weights
    data['Training_Heart_Index'] = (data['Training_Intensity'] * data['Heart_Rate_Normalized'])
    data['Recovery_Heart_Ratio'] = data['Heart_Rate_Normalized'] / (data['Recovery_Time'] + 1)
    
    # Risk assessment features with balanced weights
    data['Risk_Factor'] = (
        (data['Training_Intensity'] * 0.4) *  # Reduced training intensity weight
        (data['Heart_Rate_Status'] * 0.3) *  # Reduced heart rate weight
        (1 / (data['Recovery_Time'] + 1)) * 0.3  # Reduced recovery time weight
    )
    
    data['Health_Index'] = (
        (1 - abs(data['Heart_Rate_Normalized'] - 0.5)) * 0.4 *  # Heart rate impact
        (1 / (data['Training_Intensity'] + 1)) * 0.3 *  # Training impact
        (1 / (data['Recovery_Time'] + 1)) * 0.3  # Recovery impact
    )

    # Add new engineered features to the feature list
    features.extend(['BMI', 'Age_Normalized', 'Training_Load', 'Training_Intensity_Ratio',
                    'Heart_Rate_Normalized', 'Heart_Rate_Status',
                    'Recovery_Quality', 'Recovery_Intensity_Ratio',
                    'Training_Heart_Index', 'Recovery_Heart_Ratio',
                    'Risk_Factor', 'Health_Index'])

    X = data[features].values
    y = data[target].values

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print("Class distribution:", class_distribution)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return data, X, y, scaler, features

# Build and train the model
@st.cache_resource
def build_and_train_model(X, y, features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create the neural network model
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_prob = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = model.evaluate(X_test, y_test)[1]
    
    return model, history, accuracy, X_test, y_test, y_pred, y_prob

def plot_training_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(
        title="Model Training Performance",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        template="plotly_white"
    )
    st.plotly_chart(fig)

def show_model_metrics(X_test, y_test, y_pred, y_prob):
    st.write("### Model Performance Metrics")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Low Risk', 'High Risk'],
                    y=['Low Risk', 'High Risk'],
                    text_auto=True,
                    color_continuous_scale="RdBu")
    st.plotly_chart(fig)

    # Display classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

def main_dashboard():
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ Elite Sports Injury Prediction</h1>", unsafe_allow_html=True)
    
    # Sidebar navigation with custom styling
    st.sidebar.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Go to", ["📊 Dashboard", "🎯 Prediction", "📈 Reports", "👥 User Management"])
    
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🎯 Prediction":
        show_prediction()
    elif page == "📈 Reports":
        show_reports()
    elif page == "👥 User Management" and st.session_state.user_role == "admin":
        show_user_management()

def show_dashboard():
    st.markdown("<h2 style='text-align: center;'>📊 Performance Dashboard</h2>", unsafe_allow_html=True)
    
    data, X, y, scaler, features = load_and_preprocess_data()
    model, history, accuracy, X_test, y_test, y_pred, y_prob = build_and_train_model(X, y, features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
        st.metric("📊 Total Records", len(data))
    
    with col2:
        st.metric("👥 Average Age", f"{data['Player_Age'].mean():.1f}")
        st.metric("❤️ Average Heart Rate", f"{data['Heart_Rate'].mean():.1f}")
    
    st.markdown("<h3 style='text-align: center;'>📈 Model Performance</h3>", unsafe_allow_html=True)
    plot_training_history(history)

    # Show model metrics
    show_model_metrics(X_test, y_test, y_pred, y_prob)
    
    st.markdown("<h3 style='text-align: center;'>🔍 Feature Correlations</h3>", unsafe_allow_html=True)
    fig = px.imshow(data.corr(), text_auto=True, color_continuous_scale="RdBu")
    st.plotly_chart(fig)

def show_prediction():
    st.markdown("<h2 style='text-align: center;'>🎯 Injury Risk Assessment</h2>", unsafe_allow_html=True)
    
    # Add smartwatch connection section
    st.markdown("### ⌚ Smartwatch Integration")
    watch_type = st.selectbox("Select your smartwatch", ["None", "Fitbit", "Garmin"])
    
    if watch_type != "None":
        if not st.session_state.smartwatch_connected:
            if st.button("Connect Smartwatch"):
                if connect_smartwatch(watch_type.lower()):
                    st.session_state.smartwatch_connected = True
                    if start_heart_rate_stream(watch_type.lower()):
                        st.success("Smartwatch connected successfully!")
                    else:
                        st.error("Failed to start heart rate stream")
        else:
            st.success("Smartwatch connected")
            if st.button("Disconnect"):
                st.session_state.smartwatch_connected = False
                st.session_state.heart_rate_data = []
                st.session_state.current_heart_rate = None
                st.rerun()
    
    data, X, y, scaler, features = load_and_preprocess_data()
    model, _, _, X_test, y_test, y_pred, y_prob = build_and_train_model(X, y, features)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            player_age = st.number_input("👤 Age", 10, 50, 25)
            player_weight = st.number_input("⚖️ Weight (kg)", 30, 120, 70)
            player_height = st.number_input("📏 Height (cm)", 140, 220, 175)
            previous_injuries = st.number_input("🏥 Previous Injuries", 0, 10, 0)
        
        with col2:
            training_intensity = st.slider("💪 Training Intensity (1-10)", 1, 10, 1,
                                         help="Higher intensity means more strenuous training")
            recovery_time = st.number_input("⏰ Recovery Time (days)", 0, 30, 28,
                                          help="Time taken to recover between training sessions")
            
            # Use smartwatch heart rate if available
            if st.session_state.current_heart_rate is not None:
                heart_rate = st.number_input("❤️ Heart Rate (BPM)", 40, 200, 
                                           value=int(st.session_state.current_heart_rate),
                                           help="Real-time heart rate from smartwatch")
            else:
                heart_rate = st.number_input("❤️ Heart Rate (BPM)", 40, 200, 60,
                                           help="Normal resting heart rate is 60-100 BPM")
        
        submitted = st.form_submit_button("🚀 Predict Injury Risk")
        
        if submitted:
            # Calculate additional features with normalized values
            bmi = player_weight / ((player_height / 100) ** 2)
            age_normalized = (player_age - 15) / 35
            training_load = training_intensity * (1 / (recovery_time + 1))
            training_intensity_ratio = training_intensity / 10
            
            # Normalize heart rate to a more reasonable scale
            heart_rate_normalized = (heart_rate - 60) / 40  # Normal range is 60-100
            heart_rate_status = 0.2 if 60 <= heart_rate <= 100 else (0.4 if heart_rate < 60 else 0.8)
            
            # Calculate recovery quality with more emphasis
            recovery_quality = 1 / (recovery_time + 1)
            recovery_intensity_ratio = recovery_time / (training_intensity + 1)
            
            # Calculate combined factors with balanced weights
            training_heart_index = (training_intensity * heart_rate_normalized) * 0.3
            recovery_heart_ratio = (heart_rate_normalized / (recovery_time + 1)) * 0.3
            
            # Calculate risk factor with adjusted weights
            risk_factor = (
                (training_intensity * 0.35) *  # Increased training impact
                (heart_rate_status * 0.35) *   # Maintained heart rate importance
                (1 / (recovery_time + 1)) * 0.3  # Slightly reduced recovery importance
            )
            
            # Calculate health index with balanced weights
            health_index = (
                (1 - abs(heart_rate_normalized - 0.5)) * 0.35 *  # Heart rate impact
                (1 / (training_intensity + 1)) * 0.35 *  # Training impact
                (1 / (recovery_time + 1)) * 0.3  # Recovery impact
            )
            
            user_input = np.array([[player_age, player_weight, player_height,
                                  previous_injuries, training_intensity,
                                  recovery_time, heart_rate, bmi,
                                  age_normalized, training_load,
                                  training_intensity_ratio, heart_rate_normalized,
                                  heart_rate_status, recovery_quality,
                                  recovery_intensity_ratio, training_heart_index,
                                  recovery_heart_ratio, risk_factor, health_index]])
            
            user_input = scaler.transform(user_input)
            risk_score = model.predict(user_input)[0][0]  # Changed to use neural network prediction
            
            # Apply more balanced normalization
            normalized_risk = (risk_score - 0.05) / 0.9  # Shift and scale to get more balanced distribution
            normalized_risk = max(0, min(1, normalized_risk))  # Ensure it stays between 0 and 1
            
            # Apply additional normalization based on input factors
            if training_intensity <= 3 and 60 <= heart_rate <= 100 and recovery_time >= 7:
                normalized_risk *= 0.7  # Less aggressive reduction for conservative inputs
            elif training_intensity <= 5 and 60 <= heart_rate <= 100 and recovery_time >= 5:
                normalized_risk *= 0.8  # Less aggressive reduction for moderate inputs
            
            # Add previous injuries impact
            normalized_risk += (previous_injuries * 0.05)  # Each previous injury adds 5% to risk
            
            # Ensure risk doesn't exceed 100%
            normalized_risk = min(1.0, normalized_risk)
            
            st.markdown("<h3 style='text-align: center;'>📊 Prediction Results</h3>", unsafe_allow_html=True)
            
            # Display key factors with their individual impacts
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                    <h4>🔍 Key Factors Analysis:</h4>
                    <ul>
                        <li>Training Intensity Impact: {:.1%} (Weight: 0.35x)</li>
                        <li>Heart Rate Status: {} (Weight: 0.35x)</li>
                        <li>Recovery Quality: {:.1%} (Weight: 0.30x)</li>
                        <li>Previous Injuries Impact: +{:.1%}</li>
                        <li>Combined Risk Factor: {:.1%}</li>
                    </ul>
                </div>
            """.format(
                training_intensity_ratio,
                "Normal" if 60 <= heart_rate <= 100 else "Below Normal" if heart_rate < 60 else "Above Normal",
                recovery_quality,
                previous_injuries * 0.05,
                risk_factor
            ), unsafe_allow_html=True)
            
            # Show how each factor contributes to the risk
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                    <h4>📈 Factor Contributions:</h4>
                    <ul>
                        <li>Training Impact: {:.1%}</li>
                        <li>Heart Rate Impact: {:.1%}</li>
                        <li>Recovery Impact: {:.1%}</li>
                        <li>Previous Injuries: +{:.1%}</li>
                    </ul>
                </div>
            """.format(
                training_intensity_ratio,
                heart_rate_status,
                recovery_quality,
                previous_injuries * 0.05
            ), unsafe_allow_html=True)
            
            # Adjusted risk thresholds for more balanced distribution
            if normalized_risk > 0.75:
                st.error(f"⚠️ Very High Injury Risk ({normalized_risk:.1%})")
                st.markdown("""
                    <div style='background-color: #ffebee; padding: 20px; border-radius: 10px;'>
                        <h4>🚨 Recommendations:</h4>
                        <ul>
                            <li>Immediately reduce training intensity to level {}</li>
                            <li>Target heart rate should be between 60-100 BPM</li>
                            <li>Increase recovery time to at least {} days</li>
                            <li>Monitor heart rate closely during training</li>
                            <li>Consult a sports physician</li>
                        </ul>
                    </div>
                """.format(max(1, training_intensity - 3), min(30, recovery_time + 7)), unsafe_allow_html=True)
            elif normalized_risk > 0.60:
                st.error(f"⚠️ High Injury Risk ({normalized_risk:.1%})")
                st.markdown("""
                    <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px;'>
                        <h4>⚠️ Recommendations:</h4>
                        <ul>
                            <li>Reduce training intensity to level {}</li>
                            <li>Maintain heart rate between 60-100 BPM</li>
                            <li>Increase recovery time to {} days</li>
                            <li>Monitor heart rate closely</li>
                            <li>Consider consulting a sports physician</li>
                        </ul>
                    </div>
                """.format(max(1, training_intensity - 2), min(30, recovery_time + 5)), unsafe_allow_html=True)
            elif normalized_risk > 0.45:
                st.warning(f"⚠️ Moderate Injury Risk ({normalized_risk:.1%})")
                st.markdown("""
                    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px;'>
                        <h4>📋 Recommendations:</h4>
                        <ul>
                            <li>Maintain current training intensity</li>
                            <li>Keep heart rate between 60-100 BPM</li>
                            <li>Ensure recovery time of at least {} days</li>
                            <li>Monitor for any unusual symptoms</li>
                        </ul>
                    </div>
                """.format(max(1, recovery_time)), unsafe_allow_html=True)
            else:
                st.success(f"✅ Low Injury Risk ({normalized_risk:.1%})")
                st.markdown("""
                    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px;'>
                        <h4>💪 Recommendations:</h4>
                        <ul>
                            <li>Continue current training regimen</li>
                            <li>Maintain heart rate between 60-100 BPM</li>
                            <li>Keep good recovery practices</li>
                            <li>Monitor health indicators</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

def show_reports():
    st.markdown("<h2 style='text-align: center;'>📈 Analysis Reports</h2>", unsafe_allow_html=True)
    
    data, _, _, _, _ = load_and_preprocess_data()
    
    st.markdown("<h3 style='text-align: center;'>📊 Injury Distribution by Age</h3>", unsafe_allow_html=True)
    fig = px.histogram(data, x="Player_Age", color="Likelihood_of_Injury",
                      title="Injury Distribution by Age",
                      labels={"Player_Age": "Age", "count": "Number of Cases"})
    st.plotly_chart(fig)
    
    st.markdown("<h3 style='text-align: center;'>💪 Training Intensity vs Injury Risk</h3>", unsafe_allow_html=True)
    fig = px.scatter(data, x="Training_Intensity", y="Likelihood_of_Injury",
                    color="Player_Age", size="Heart_Rate",
                    title="Training Intensity vs Injury Risk",
                    labels={"Training_Intensity": "Training Intensity",
                           "Likelihood_of_Injury": "Injury Risk"})
    st.plotly_chart(fig)

def show_user_management():
    st.markdown("<h2 style='text-align: center;'>👥 User Management</h2>", unsafe_allow_html=True)
    
    users = load_users()
    
    st.markdown("<h3 style='text-align: center;'>✨ Add New User</h3>", unsafe_allow_html=True)
    new_username = st.text_input("👤 Username")
    new_password = st.text_input("🔒 Password", type="password")
    new_role = st.selectbox("🎯 Role", ["athlete", "coach", "admin"])
    
    if st.button("🚀 Add User"):
        if new_username and new_password:
            users[new_username] = {
                'password': hashlib.sha256(new_password.encode()).hexdigest(),
                'role': new_role
            }
            save_users(users)
            st.success("✅ User added successfully!")
        else:
            st.error("❌ Please fill in all fields")
    
    st.markdown("<h3 style='text-align: center;'>📋 Current Users</h3>", unsafe_allow_html=True)
    user_df = pd.DataFrame([
        {"Username": username, "Role": data['role']}
        for username, data in users.items()
    ])
    st.dataframe(user_df)

def connect_smartwatch(watch_type):
    """Connect to the specified smartwatch type"""
    if watch_type == 'fitbit':
        return connect_fitbit()
    elif watch_type == 'garmin':
        return connect_garmin()
    else:
        st.error("Unsupported smartwatch type")
        return False

def connect_fitbit():
    """Connect to Fitbit API"""
    try:
        # Implement Fitbit OAuth2 authentication
        auth_url = f"https://www.fitbit.com/oauth2/authorize?response_type=code&client_id={SMARTWATCH_CONFIG['fitbit']['client_id']}&redirect_uri={SMARTWATCH_CONFIG['fitbit']['redirect_uri']}&scope=heartrate"
        st.markdown(f"[Connect Fitbit]({auth_url})")
        return True
    except Exception as e:
        st.error(f"Error connecting to Fitbit: {str(e)}")
        return False

def connect_garmin():
    """Connect to Garmin API"""
    try:
        # Implement Garmin OAuth1 authentication
        # This is a placeholder for Garmin's OAuth implementation
        st.info("Garmin connection not implemented yet")
        return False
    except Exception as e:
        st.error(f"Error connecting to Garmin: {str(e)}")
        return False

def start_heart_rate_stream(watch_type):
    """Start streaming heart rate data"""
    if watch_type == 'fitbit':
        return start_fitbit_stream()
    elif watch_type == 'garmin':
        return start_garmin_stream()
    return False

def start_fitbit_stream():
    """Start streaming heart rate data from Fitbit"""
    try:
        # Create a WebSocket connection to Fitbit's streaming API
        ws = websocket.WebSocketApp(
            "wss://api.fitbit.com/1/user/-/activities/heart.json",
            on_message=on_heart_rate_message,
            on_error=on_error,
            on_close=on_close
        )
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        return True
    except Exception as e:
        st.error(f"Error starting Fitbit stream: {str(e)}")
        return False

def on_heart_rate_message(ws, message):
    """Handle incoming heart rate messages"""
    try:
        data = json.loads(message)
        if 'heartRate' in data:
            st.session_state.heart_rate_data.append({
                'timestamp': datetime.now(),
                'heart_rate': data['heartRate']
            })
            st.session_state.current_heart_rate = data['heartRate']
    except Exception as e:
        st.error(f"Error processing heart rate data: {str(e)}")

def on_error(ws, error):
    """Handle WebSocket errors"""
    st.error(f"WebSocket error: {str(error)}")

def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket connection close"""
    st.info("Heart rate stream closed")

def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
