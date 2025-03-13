import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('injury_data.csv')

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data.iloc[:, :] = imputer.fit_transform(data)

    # Select features and target
    features = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries',
                'Training_Intensity', 'Recovery_Time', 'Heart_Rate']
    target = 'Likelihood_of_Injury'

    X = data[features].values
    y = data[target].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return data, X, y, scaler

# Build and train the model
@st.cache_resource
def build_and_train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return model, history, accuracy

# Visualize training history
def plot_training_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(title="Loss Over Epochs", xaxis_title="Epochs", yaxis_title="Loss")
    st.plotly_chart(fig)

# Streamlit Web Application
def main():
    st.title("🏀 Sports Injury Prediction App")
    st.write("Predict the likelihood of sports injuries using deep learning.")

    data, X, y, scaler = load_and_preprocess_data()
    model, history, accuracy = build_and_train_model(X, y)

    st.sidebar.title("⚙️ Input Athlete Data")
    player_age = st.sidebar.slider("Age", 10, 50, 25, step=1)
    player_weight = st.sidebar.slider("Weight (kg)", 30, 120, 70, step=1)
    player_height = st.sidebar.slider("Height (cm)", 140, 220, 175, step=1)
    previous_injuries = st.sidebar.slider("Previous Injuries", 0, 10, 1, step=1)
    training_intensity = st.sidebar.slider("Training Intensity (1-10)", 1, 10, 5, step=1)
    recovery_time = st.sidebar.slider("Recovery Time (days)", 0, 30, 7, step=1)
    heart_rate = st.sidebar.slider("Heart Rate (BPM)", 40, 200, 80, step=1)

    if st.sidebar.button("🚀 Predict Injury Risk"):
        user_input = np.array([[player_age, player_weight, player_height,
                                previous_injuries, training_intensity,
                                recovery_time, heart_rate]])
        user_input = scaler.transform(user_input)

        prediction = model.predict(user_input)
        prediction = (prediction > 0.5).astype(int)

        if prediction[0][0] == 1:
            st.error("❌ High Injury Risk")
        else:
            st.success("✅ Low Injury Risk")

    st.write("### 📊 Dataset Insights")
    st.dataframe(data.head(10))
    st.write(f"**Model Accuracy:** {accuracy:.2%}")

    st.write("### 📈 Model Training Performance")
    plot_training_history(history)

    st.write("### 🔍 Correlation Heatmap")
    fig = px.imshow(data.corr(), text_auto=True, color_continuous_scale="RdBu")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
