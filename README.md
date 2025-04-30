# Final year project
Roll number - 10201100096
This is a sports injury prediction application software that help users (athletes and coaches especially) predict the likelihood of injury based on certain key factors like heart rate and training intensity, in order to reduce damage to the team an individual

Overview
This project implements an advanced injury prediction system for athletes using machine learning. The system analyzes various factors including heart rate, training intensity, and recovery time to predict injury risk and provide personalized recommendations.

Features
- Real-time injury risk assessment
- Smartwatch integration (Fitbit & Garmin)
- Personalized training recommendations
- User authentication system
- Interactive dashboard with performance metrics
- Detailed analysis reports

Technical Stack
- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Imbalanced-learn

Installation
1. Clone the repository:
```bash
git clone https://github.com/Afrikind/Injury_Prediction.git
cd Injury_Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run Injury_Prediction_app.py
```

 Project Structure
```
Injury_Prediction/
├── Injury_Prediction_app.py    # Main application file
├── injury_data.csv            # Dataset
├── requirements.txt           # Dependencies
├── users.json                # User authentication data
└── README.md                 # Documentation
```

 Usage
1. Launch the application
2. Create an account or login
3. Navigate through the dashboard
4. Input athlete data for risk assessment
5. View detailed analysis and recommendations

 Model Details
- Algorithm: Random Forest Classifier
- Key Features: Training Intensity, Heart Rate, Recovery Time
- Risk Assessment: Multi-level classification
- Performance Metrics: Accuracy, Precision, Recall

